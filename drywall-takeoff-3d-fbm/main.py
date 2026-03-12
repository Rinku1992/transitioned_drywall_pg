import os
import sys
import re
import uuid
import time as time_module
from datetime import timedelta, datetime, date, time
from decimal import Decimal
from base64 import b64encode
from pathlib import Path
import json
from time import time as from_unix_epoch
from collections import defaultdict
import asyncio
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pydantic_core import ValidationError
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import math

from preprocessing import preprocess
from extrapolate_3d import Extrapolate3D
from helper import (
    create_pg_pool,
    pg_execute,
    pg_fetch_one,
    pg_fetch_all,
    log_json,
    timed_step,
    parse_jsonb,
    get_gcs_client,
    load_gcp_credentials,
    load_hyperparameters,
    sha256,
    upload_floorplan,
    insert_model_2d,
    is_duplicate,
    delete_plan,
    load_floorplan_to_structured_2d_ID_token,
    load_vertex_ai_client,
    classify_plan,
)


# ---------------------------------------------------------------------------
# Validation & Response Helpers
# ---------------------------------------------------------------------------

def respond_with_UI_payload(payload, status_code=200):
    """Return a JSON response. Uses jsonable_encoder to handle datetime/Decimal."""
    return JSONResponse(
        content=jsonable_encoder(payload),
        status_code=status_code,
        media_type="application/json",
    )


def validate_required(params: dict, required_fields: list, endpoint: str, rid: str):
    """Validate required fields exist and are non-empty.
    Returns (True, None) if valid, (False, JSONResponse) if invalid.
    """
    missing = [f for f in required_fields if params.get(f) is None]
    if missing:
        log_json("WARNING", "VALIDATION_FAILED", request_id=rid, endpoint=endpoint,
                 missing_fields=missing)
        return False, respond_with_UI_payload(
            dict(error=f"Missing required fields: {', '.join(missing)}"),
            status_code=400
        )
    return True, None


def require_pool(pool, endpoint: str, rid: str):
    """Check if PG pool is available. Returns error response if not."""
    if pool is None:
        log_json("ERROR", "POOL_UNAVAILABLE", request_id=rid, endpoint=endpoint)
        return respond_with_UI_payload(
            dict(error="Database unavailable. Please try again later."),
            status_code=503
        )
    return None


# ---------------------------------------------------------------------------
# GCS Helper (uses shared client from helper.py)
# ---------------------------------------------------------------------------

def download_floorplan(plan_id, project_id, credentials, destination_path="/tmp/floor_plan.PDF"):
    client = get_gcs_client()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    blob.download_to_filename(destination_path)
    return f"gs://{credentials['CloudStorage']['bucket_name']}/{blob_path}"


def floorplan_to_structured_2d(credentials, id_token, project_id, plan_id, user_id, page_number):
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }
    try:
        requests.post(
            f"{credentials['CloudRun']['APIs']['floorplan_to_structured_2d']}/floorplan_to_structured_2d",
            headers=headers,
            json=dict(
                project_id=project_id,
                plan_id=plan_id,
                user_id=user_id,
                page_number=page_number
            ),
            timeout=1800,
        )
    except Exception as e:
        log_json("WARNING", "FLOORPLAN_TO_2D_CALL_FAILED",
                 page_number=page_number, error=str(e))
        

def get_params(request_query_params, body):
    """Merge query params and body into a single dict. Query params take priority."""
    merged = dict(body) if body else {}
    merged.update(dict(request_query_params))
    return merged


# ---------------------------------------------------------------------------
# Database Operations (migrated from BigQuery inline functions)
# ---------------------------------------------------------------------------

async def insert_project(payload_project, pool, credentials, rid=""):
    """Insert a new project if it doesn't already exist. Returns created_at."""
    async with timed_step("insert_project", request_id=rid, project_id=payload_project.project_id):
        await pg_execute(
            pool,
            """
            INSERT INTO projects (
                project_id, project_name, project_location, fbm_branch,
                project_type, project_area, contractor_name, created_at, created_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), $8)
            ON CONFLICT (LOWER(project_id)) DO NOTHING
            """,
            [
                payload_project.project_id, payload_project.project_name,
                payload_project.project_location, payload_project.FBM_branch,
                payload_project.project_type, payload_project.project_area,
                payload_project.contractor_name, payload_project.created_by
            ],
            query_name="insert_project"
        )
        row = await pg_fetch_one(
            pool,
            "SELECT created_at FROM projects WHERE LOWER(project_id) = LOWER($1)",
            [payload_project.project_id],
            query_name="insert_project__get_created_at"
        )
        return row["created_at"].isoformat() if row else None


async def insert_plan(
    project_id, user_id, status, pool, credentials,
    payload_plan=None, plan_id=None, size_in_bytes=None,
    GCS_URL_floorplan=None, n_pages=None, sha_256=None
):
    """Upsert a plan row. BQ MERGE → PG INSERT ... ON CONFLICT.
    Now accepts sha_256 as param to avoid redundant PDF download.
    """
    if sha_256 is None:
        sha_256 = ''
    if plan_id and not sha_256:
        # Only download + hash if sha_256 not provided by caller
        pdf_path = Path("/tmp/floor_plan.PDF")
        download_floorplan(plan_id, project_id, credentials, destination_path=pdf_path)
        sha_256 = sha256(pdf_path)
    if not plan_id:
        plan_id = payload_plan.plan_id
    plan_name, plan_type, file_type = '', '', ''
    if payload_plan:
        plan_name = payload_plan.plan_name
        plan_type = payload_plan.plan_type
        file_type = payload_plan.file_type
    if not n_pages:
        n_pages = 0
    if not GCS_URL_floorplan:
        GCS_URL_floorplan = ''
    if not size_in_bytes:
        size_in_bytes = 0

    await pg_execute(
        pool,
        """
        INSERT INTO plans (
            plan_id, project_id, user_id, status, plan_name, plan_type,
            file_type, pages, size_in_bytes, source, sha256, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW())
        ON CONFLICT (LOWER(project_id), LOWER(plan_id))
        DO UPDATE SET
            pages = EXCLUDED.pages,
            source = EXCLUDED.source,
            sha256 = EXCLUDED.sha256,
            status = EXCLUDED.status,
            size_in_bytes = EXCLUDED.size_in_bytes,
            user_id = EXCLUDED.user_id,
            updated_at = NOW()
        """,
        [plan_id, project_id, user_id, status, plan_name, plan_type,
         file_type, n_pages, size_in_bytes, GCS_URL_floorplan, sha_256],
        query_name="insert_plan"
    )


async def insert_model_2d_revision(
    model_2d, scale, page_number, plan_id, user_id, project_id, pool, credentials
):
    """Insert a new 2D model revision, auto-incrementing revision_number."""
    page_number = int(page_number)
    model_2d_json = json.dumps(model_2d)

    if not model_2d.get("metadata", None):
        row = await pg_fetch_one(
            pool,
            "SELECT model_2d->'metadata' AS metadata FROM models "
            "WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
            [project_id, plan_id, page_number],
            query_name="insert_model_2d_revision__fetch_metadata"
        )
        if row:
            metadata = parse_jsonb(row["metadata"])
            if metadata:
                model_2d["metadata"] = metadata
                model_2d_json = json.dumps(model_2d)

    row = await pg_fetch_one(
        pool,
        "SELECT MAX(revision_number) AS revision_number FROM model_revisions_2d "
        "WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, page_number],
        query_name="insert_model_2d_revision__max_rev"
    )
    revision_number = (row["revision_number"] or 0) + 1 if row else 1

    await pg_execute(
        pool,
        """
        INSERT INTO model_revisions_2d (
            plan_id, project_id, user_id, page_number, scale, model, created_at, revision_number
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW(), $7)
        """,
        [plan_id, project_id, user_id, page_number, scale or '', model_2d_json, revision_number],
        query_name="insert_model_2d_revision"
    )


async def insert_model_3d(
    model_3d, scale, page_number, plan_id, user_id, project_id, pool, credentials
):
    """Update the model_3d column in the models table."""
    page_number = int(page_number)
    model_3d_json = json.dumps(model_3d)
    scale = scale or ''

    await pg_execute(
        pool,
        """
        UPDATE models SET
            model_3d = $1::jsonb,
            scale = CASE WHEN $2 = '' THEN scale ELSE $2 END,
            user_id = $3,
            updated_at = NOW()
        WHERE LOWER(project_id) = LOWER($4)
          AND LOWER(plan_id) = LOWER($5)
          AND page_number = $6
        """,
        [model_3d_json, scale, user_id, project_id, plan_id, page_number],
        query_name="insert_model_3d"
    )


async def insert_model_3d_revision(
    model_3d, scale, page_number, plan_id, user_id, project_id, pool, credentials
):
    """Insert a new 3D model revision, auto-incrementing revision_number."""
    page_number = int(page_number)
    model_3d_json = json.dumps(model_3d)

    row = await pg_fetch_one(
        pool,
        "SELECT MAX(revision_number) AS revision_number FROM model_revisions_3d "
        "WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, page_number],
        query_name="insert_model_3d_revision__max_rev"
    )
    revision_number = (row["revision_number"] or 0) + 1 if row else 1

    await pg_execute(
        pool,
        """
        INSERT INTO model_revisions_3d (
            plan_id, project_id, user_id, page_number, scale,
            model, takeoff, created_at, revision_number
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, '{}'::jsonb, NOW(), $7)
        """,
        [plan_id, project_id, user_id, page_number, scale or '', model_3d_json, revision_number],
        query_name="insert_model_3d_revision"
    )


async def insert_takeoff(
    takeoff, page_number, plan_id, user_id, project_id,
    revision_number, pool, credentials
):
    """Update takeoff in models table, and optionally in revisions."""
    page_number = int(page_number)
    takeoff_json = json.dumps(takeoff)

    await pg_execute(
        pool,
        """
        UPDATE models SET
            takeoff = $1::jsonb,
            updated_at = NOW(),
            user_id = $2
        WHERE LOWER(project_id) = LOWER($3)
          AND LOWER(plan_id) = LOWER($4)
          AND page_number = $5
        """,
        [takeoff_json, user_id, project_id, plan_id, page_number],
        query_name="insert_takeoff__models"
    )

    if revision_number:
        revision_number = int(revision_number)
        await pg_execute(
            pool,
            """
            UPDATE model_revisions_3d SET
                takeoff = $1::jsonb,
                user_id = $2
            WHERE LOWER(project_id) = LOWER($3)
              AND LOWER(plan_id) = LOWER($4)
              AND page_number = $5
              AND revision_number = $6
            """,
            [takeoff_json, user_id, project_id, plan_id, page_number, revision_number],
            query_name="insert_takeoff__revision_3d"
        )


async def delete_floorplan(project_id, plan_id, user_id, pool, credentials):
    """Delete a plan and all related models/revisions, then clean up GCS.
    Fixed: GCS prefix uses project_id/plan_id/ (not user_id — blobs aren't stored under user_id).
    """
    params_3 = [project_id, plan_id, user_id]
    params_2 = [project_id, plan_id]

    await pg_execute(pool,
        "DELETE FROM model_revisions_3d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        params_2, query_name="delete_floorplan__rev_3d")
    await pg_execute(pool,
        "DELETE FROM model_revisions_2d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        params_2, query_name="delete_floorplan__rev_2d")
    await pg_execute(pool,
        "DELETE FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        params_2, query_name="delete_floorplan__models")
    await pg_execute(pool,
        "DELETE FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND LOWER(user_id) = LOWER($3)",
        params_3, query_name="delete_floorplan__plans")

    # GCS cleanup — blobs are stored under project_id/plan_id/ (no user_id in path)
    client = get_gcs_client()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    prefix = f"{project_id.lower()}/{plan_id.lower()}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)
        log_json("INFO", "GCS_CLEANUP", prefix=prefix, blobs_deleted=len(blobs))


# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Drywall Takeoff (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
HYPERPARAMETERS = load_hyperparameters()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Module-level singletons (created once, never re-created) ---
pg_pool = None
VERTEX_AI_CLIENT = None
VERTEX_AI_GENERATION_CONFIG = None
VERTEX_AI_MAX_RETRY = None


@app.on_event("startup")
async def startup():
    global pg_pool, VERTEX_AI_CLIENT, VERTEX_AI_GENERATION_CONFIG, VERTEX_AI_MAX_RETRY

    # 1. PostgreSQL pool
    try:
        pg_pool = await create_pg_pool(CREDENTIALS)
        if pg_pool:
            log_json("INFO", "STARTUP_PG_SUCCESS", detail="PostgreSQL pool created")
        else:
            log_json("WARNING", "STARTUP_PG_DEGRADED", detail="PostgreSQL pool is None")
    except Exception as exc:
        log_json("ERROR", "STARTUP_PG_FAILED", error=f"{type(exc).__name__}: {exc}")
        pg_pool = None

    # 2. Vertex AI client (loaded once, reused for all requests)
    try:
        VERTEX_AI_CLIENT, VERTEX_AI_GENERATION_CONFIG = load_vertex_ai_client(CREDENTIALS)
        VERTEX_AI_MAX_RETRY = CREDENTIALS["VertexAI"]["llm"]["max_retry"]
        log_json("INFO", "STARTUP_VERTEX_AI_SUCCESS",
                 model=CREDENTIALS["VertexAI"]["llm"]["model_name"])
    except Exception as exc:
        log_json("ERROR", "STARTUP_VERTEX_AI_FAILED", error=f"{type(exc).__name__}: {exc}")

    # 3. GCS client (initialize singleton)
    get_gcs_client()

    log_json("INFO", "STARTUP_COMPLETE")


@app.on_event("shutdown")
async def shutdown():
    global pg_pool
    if pg_pool:
        await pg_pool.close()
        log_json("INFO", "SHUTDOWN", detail="PostgreSQL pool closed")


class PayloadProject(BaseModel):
    project_id: str
    project_name: str
    project_location: str
    project_area: str
    project_type: str
    contractor_name: str
    FBM_branch: str
    created_by: str


class PayloadPlan(BaseModel):
    plan_id: str
    plan_name: str
    plan_type: str
    file_type: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate_project")
async def generate_project(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/generate_project")

    pool_err = require_pool(pg_pool, "/generate_project", rid)
    if pool_err:
        return pool_err

    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    try:
        payload_project = PayloadProject(**parameters)
    except (ValidationError, Exception):
        try:
            payload_project = PayloadProject(**body)
        except (ValidationError, Exception) as e:
            log_json("WARNING", "VALIDATION_FAILED", request_id=rid,
                     endpoint="/generate_project", error=str(e))
            return respond_with_UI_payload(
                dict(error=f"Invalid project payload: {e}"), status_code=400
            )

    created_at = await insert_project(payload_project, pg_pool, CREDENTIALS, rid=rid)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/generate_project",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             project_id=payload_project.project_id)
    return respond_with_UI_payload(
        dict(
            project_id=payload_project.project_id,
            project_name=payload_project.project_name,
            created_at=created_at,
        )
    )


@app.post("/load_projects")
async def load_projects(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_projects")

    pool_err = require_pool(pg_pool, "/load_projects", rid)
    if pool_err:
        return pool_err

    rows = await pg_fetch_all(pg_pool, "SELECT * FROM projects", query_name="load_projects")
    projects = [dict(row) for row in rows]

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_projects",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             project_count=len(projects))
    return respond_with_UI_payload(jsonable_encoder({"projects": projects}))


@app.post("/load_project_plans")
async def load_project_plans(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_project_plans")

    pool_err = require_pool(pg_pool, "/load_project_plans", rid)
    if pool_err:
        return pool_err

    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id"], "/load_project_plans", rid)
    if not valid:
        return err
    project_id = params["project_id"]

    project_row = await pg_fetch_one(
        pg_pool,
        "SELECT * FROM projects WHERE LOWER(project_id) = LOWER($1)",
        [project_id],
        query_name="load_project_plans__project"
    )
    if not project_row:
        return respond_with_UI_payload(dict(project_metadata=dict(), project_plans=list()))

    project_metadata = dict(project_row)

    plan_rows = await pg_fetch_all(
        pg_pool,
        "SELECT * FROM plans WHERE LOWER(project_id) = LOWER($1)",
        [project_id],
        query_name="load_project_plans__plans"
    )
    project_plans = [dict(row) for row in plan_rows]

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_project_plans",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             project_id=project_id, plan_count=len(project_plans))
    return respond_with_UI_payload(
        jsonable_encoder({"project_metadata": project_metadata, "project_plans": project_plans})
    )


@app.post("/generate_floorplan_upload_signed_URL")
async def generate_floorplan_upload_signed_URL(request: Request) -> str:
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/generate_floorplan_upload_signed_URL")

    pool_err = require_pool(pg_pool, "/generate_floorplan_upload_signed_URL", rid)
    if pool_err:
        return pool_err

    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "user_id", "plan"], "/generate_floorplan_upload_signed_URL", rid)
    if not valid:
        return err

    project_id = params["project_id"]
    user_id = params["user_id"]
    payload_plan = PayloadPlan(**params["plan"])

    await insert_plan(project_id, user_id, "NOT STARTED", pg_pool, CREDENTIALS, payload_plan=payload_plan)

    client = get_gcs_client()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{payload_plan.plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="PUT",
        content_type="application/octet-stream",
    )
    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/generate_floorplan_upload_signed_URL",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return url


@app.post("/load_plan_pages")
async def load_plan_pages(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_plan_pages")

    pool_err = require_pool(pg_pool, "/load_plan_pages", rid)
    if pool_err:
        return pool_err

    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id"], "/load_plan_pages", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    plan_id = params["plan_id"]

    rows = await pg_fetch_all(
        pg_pool,
        "SELECT * FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        [project_id, plan_id],
        query_name="load_plan_pages"
    )
    records = [dict(row) for row in rows]

    plan_metadata = records[0] if records else dict()
    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_plan_pages",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             page_count=len(records))
    return respond_with_UI_payload(dict(plan_metadata=plan_metadata, plan_pages=records))


@app.post("/floorplan_to_2d")
async def floorplan_to_2d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/floorplan_to_2d")

    pool_err = require_pool(pg_pool, "/floorplan_to_2d", rid)
    if pool_err:
        return pool_err

    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id"], "/floorplan_to_2d", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    user_id = params["user_id"]
    plan_id = params["plan_id"]

    pdf_path = Path("/tmp/floor_plan.PDF")
    async with timed_step("download_floorplan", request_id=rid, plan_id=plan_id):
        GCS_URL_floorplan = download_floorplan(plan_id, project_id, CREDENTIALS, destination_path=pdf_path)

    # Compute sha256 once here, pass to insert_plan to avoid redundant download
    sha_256 = sha256(pdf_path)

    async with timed_step("is_duplicate_check", request_id=rid, plan_id=plan_id):
        plan_duplicate = await is_duplicate(pg_pool, CREDENTIALS, pdf_path, project_id)
    if plan_duplicate:
        await delete_plan(pg_pool, CREDENTIALS, plan_id, project_id)
        log_json("WARNING", "DUPLICATE_PLAN", request_id=rid, plan_id=plan_id)
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))

    client = get_gcs_client()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"tmp/{user_id.lower()}/{project_id.lower()}/{plan_id.lower()}/floorplan_structured_2d.json"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.delete()

    size_in_bytes = Path(pdf_path).stat().st_size
    async with timed_step("preprocess_pdf", request_id=rid, plan_id=plan_id):
        floor_plan_paths_vector, floor_plan_paths_preprocessed = preprocess(pdf_path)

    await insert_plan(
        project_id, user_id, "IN PROGRESS", pg_pool, CREDENTIALS,
        plan_id=plan_id, size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
        sha_256=sha_256,
    )
    log_json("INFO", "STEP_COMPLETE", request_id=rid, step="insert_plan_in_progress",
             n_pages=len(floor_plan_paths_preprocessed))

    walls_2d_all = dict(pages=list())
    status = "COMPLETED"
    # Use module-level cached Vertex AI client
    vertex_ai_client_parameters = (VERTEX_AI_CLIENT, VERTEX_AI_GENERATION_CONFIG, VERTEX_AI_MAX_RETRY)
    
    # ---> HELPER FUNCTION FOR PARALLEL CLASSIFICATION & UPLOAD
    def process_single_page(index, floor_plan_vector, floor_plan_path, creds):
        p_type = classify_plan(floor_plan_path, vertex_ai_client_parameters)
        baseline_src = upload_floorplan(floor_plan_vector, plan_id, project_id, creds, index=str(index).zfill(2))
        page_src = upload_floorplan(floor_plan_path, plan_id, project_id, creds, index=str(index).zfill(2))
        return p_type, baseline_src, page_src

    try:
        id_token = load_floorplan_to_structured_2d_ID_token(CREDENTIALS)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 1. Submit all pages for parallel classification & upload
            futures = list()
            for index, (floor_plan_vector, floor_plan_path) in enumerate(zip(floor_plan_paths_vector, floor_plan_paths_preprocessed)):
                futures.append(
                    executor.submit(
                        process_single_page,
                        index, floor_plan_vector, floor_plan_path, CREDENTIALS
                    )
                )

            # Wait for all background classification threads to finish
            results = [future.result() for future in futures]
            
            # Unpack the results cleanly
            plan_types = [r[0] for r in results]
            floorplan_baseline_page_sources = [r[1] for r in results]
            floorplan_page_sources = [r[2] for r in results]

            # 2. Trigger downstream API requests in the background ONLY for valid floorplans
            for index, plan_type in enumerate(plan_types):
                if plan_type["plan_type"].upper().find("FLOOR") != -1:
                    executor.submit(
                        floorplan_to_structured_2d,
                        CREDENTIALS, id_token, project_id, plan_id, user_id, index
                    )

            # 3. Enter the DB Polling Loop
            for page_number, (plan_type, _, floorplan_page_source) in enumerate(zip(plan_types, floorplan_baseline_page_sources, floorplan_page_sources)):
                if plan_type["plan_type"].upper().find("FLOOR") == -1:
                    continue
                timeout = from_unix_epoch() + 3600
    # status = "COMPLETED"
    # # Use module-level cached Vertex AI client
    # vertex_ai_client_parameters = (VERTEX_AI_CLIENT, VERTEX_AI_GENERATION_CONFIG, VERTEX_AI_MAX_RETRY)
    # try:
    #     id_token = load_floorplan_to_structured_2d_ID_token(CREDENTIALS)
    #     with ThreadPoolExecutor(max_workers=3) as executor:
    #         futures = list()
    #         floorplan_baseline_page_sources = list()
    #         floorplan_page_sources = list()
    #         plan_types = list()
    #         for index, (floor_plan_vector, floor_plan_path) in enumerate(zip(floor_plan_paths_vector, floor_plan_paths_preprocessed)):
    #             plan_type = classify_plan(floor_plan_path, vertex_ai_client_parameters)
    #             plan_types.append(plan_type)
    #             floorplan_baseline_page_source = upload_floorplan(floor_plan_vector, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
    #             floorplan_baseline_page_sources.append(floorplan_baseline_page_source)
    #             floorplan_page_source = upload_floorplan(floor_plan_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
    #             floorplan_page_sources.append(floorplan_page_source)
    #             if plan_type["plan_type"].upper().find("FLOOR") == -1:
    #                 continue
    #             futures.append(
    #                 executor.submit(
    #                     floorplan_to_structured_2d,
    #                     CREDENTIALS, id_token, project_id, plan_id, user_id, index
    #                 )
    #             )
    #         for page_number, (plan_type, _, floorplan_page_source) in enumerate(zip(plan_types, floorplan_baseline_page_sources, floorplan_page_sources)):
    #             if plan_type["plan_type"].upper().find("FLOOR") == -1:
    #                 continue
    #             timeout = from_unix_epoch() + 3600
                poll_count = 0
                while from_unix_epoch() < timeout:
                    query_output = await pg_fetch_all(
                        pg_pool,
                        "SELECT scale, model_2d FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
                        [project_id, plan_id, page_number],
                        query_name="floorplan_to_2d__poll_model"
                    )
                    poll_count += 1
                    if query_output:
                        break
                    await asyncio.sleep(2)
                log_json("INFO", "POLL_COMPLETE", request_id=rid, step="poll_2d_model",
                         page_number=page_number, poll_iterations=poll_count)

                model_2d_raw = query_output[0]["model_2d"]
                walls_2d = parse_jsonb(model_2d_raw)
                if not walls_2d or not walls_2d.get("polygons") or not walls_2d.get("walls_2d"):
                    await pg_execute(
                        pg_pool,
                        "DELETE FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
                        [project_id, plan_id, page_number],
                        query_name="floorplan_to_2d__delete_empty_model"
                    )
                    continue
                await pg_execute(
                    pg_pool,
                    "UPDATE models SET source = $1 WHERE LOWER(project_id) = LOWER($2) AND LOWER(plan_id) = LOWER($3) AND page_number = $4",
                    [floorplan_page_source, project_id, plan_id, page_number],
                    query_name="floorplan_to_2d__update_source"
                )
                page = dict(
                    plan_id=plan_id,
                    page_number=page_number,
                    page_type=plan_type["plan_type"].upper(),
                    scale=query_output[0]["scale"],
                    walls_2d=walls_2d["walls_2d"],
                    polygons=walls_2d["polygons"],
                    **walls_2d.get("metadata", dict())
                )
                walls_2d_all["pages"].append(page)
    except Exception as e:
        log_json("ERROR", "FLOORPLAN_EXTRACTION_FAILED", request_id=rid, error=str(e))
        status = "FAILED"

    await insert_plan(
        project_id, user_id, status, pg_pool, CREDENTIALS,
        plan_id=plan_id, size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
        sha_256=sha_256,
    )

    with open("/tmp/floorplan_structured_2d.json", 'w') as f:
        json.dump(walls_2d_all, f, indent=4)
    blob.upload_from_filename("/tmp/floorplan_structured_2d.json")

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/floorplan_to_2d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             status=status, page_count=len(walls_2d_all["pages"]))
    return respond_with_UI_payload(walls_2d_all)


@app.post("/load_2d_revision")
async def load_2d_revision(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_2d_revision")

    pool_err = require_pool(pg_pool, "/load_2d_revision", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number", "revision_number"], "/load_2d_revision", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    page_number = int(params["page_number"])
    revision_number = int(params["revision_number"])

    row = await pg_fetch_one(
        pg_pool,
        "SELECT model FROM model_revisions_2d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3 AND revision_number = $4",
        [project_id, plan_id, page_number, revision_number],
        query_name="load_2d_revision"
    )
    walls_2d_JSON = parse_jsonb(row["model"]) if row else dict()

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_2d_revision",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(walls_2d_JSON)


@app.post("/load_available_revision_numbers_2d")
async def load_available_revision_numbers_2d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_available_revision_numbers_2d")

    pool_err = require_pool(pg_pool, "/load_available_revision_numbers_2d", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number"], "/load_available_revision_numbers_2d", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    page_number = int(params["page_number"])

    rows = await pg_fetch_all(
        pg_pool,
        "SELECT revision_number FROM model_revisions_2d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, page_number],
        query_name="load_available_revision_numbers_2d"
    )
    revision_numbers = [row["revision_number"] for row in rows if row["revision_number"] is not None]

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_available_revision_numbers_2d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             revision_count=len(revision_numbers))
    return respond_with_UI_payload(revision_numbers)


@app.post("/load_2d_all")
async def load_2d_all(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_2d_all")

    pool_err = require_pool(pg_pool, "/load_2d_all", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id"], "/load_2d_all", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    page_number = params.get("page_number", '')

    # Wait for plan processing to complete
    plan_row = await pg_fetch_one(
        pg_pool,
        "SELECT pages, status FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        [project_id, plan_id],
        query_name="load_2d_all__get_plan"
    )
    if not plan_row:
        return respond_with_UI_payload(dict(error="Floor Plan does not exist"))

    n_pages = plan_row["pages"]
    status = plan_row["status"]
    timeout = from_unix_epoch() + (n_pages * 120)
    poll_count = 0
    while status != "COMPLETED" and from_unix_epoch() < timeout:
        await asyncio.sleep(2)
        status_row = await pg_fetch_one(
            pg_pool,
            "SELECT status FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
            [project_id, plan_id],
            query_name="load_2d_all__poll_status"
        )
        poll_count += 1
        if not status_row:
            return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
        status = status_row["status"]

    log_json("INFO", "POLL_COMPLETE", request_id=rid, step="poll_plan_status",
             poll_iterations=poll_count, final_status=status)

    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed within timeout"), status_code=500)

    walls_2d_all = dict(pages=list())
    if page_number != '':
        page_number = int(page_number)
        rows = await pg_fetch_all(
            pg_pool,
            "SELECT page_number, scale, model_2d FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3 ORDER BY page_number",
            [project_id, plan_id, page_number],
            query_name="load_2d_all__by_page"
        )
    else:
        rows = await pg_fetch_all(
            pg_pool,
            "SELECT page_number, scale, model_2d FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) ORDER BY page_number",
            [project_id, plan_id],
            query_name="load_2d_all__all_pages"
        )

    for row in rows:
        if not row["model_2d"]:
            continue
        walls_2d = parse_jsonb(row["model_2d"])
        if not walls_2d:
            continue
        page = {
            "plan_id": plan_id,
            "page_number": row["page_number"],
            "scale": row["scale"],
            "walls_2d": walls_2d.get("walls_2d", list()),
            "polygons": walls_2d.get("polygons", list()),
            **walls_2d.get("metadata", dict()),
        }
        walls_2d_all["pages"].append(page)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_2d_all",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             page_count=len(walls_2d_all["pages"]))
    return respond_with_UI_payload(walls_2d_all)


@app.post("/update_floorplan_to_2d")
async def update_floorplan_to_2d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/update_floorplan_to_2d")

    pool_err = require_pool(pg_pool, "/update_floorplan_to_2d", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id", "page_number"], "/update_floorplan_to_2d", rid)
    if not valid:
        return err

    walls_2d_JSON = params.get("walls_2d")
    polygons_JSON = params.get("polygons")
    scale = params.get("scale")
    project_id = params["project_id"]
    user_id = params["user_id"]
    plan_id = params["plan_id"]
    index = params["page_number"]

    await insert_model_2d(
        dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON),
        scale, index, plan_id, user_id, project_id, None, None, pg_pool, CREDENTIALS
    )
    await insert_model_2d_revision(
        dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON),
        scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS
    )
    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/update_floorplan_to_2d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(status="success"))


@app.post("/update_scale")
async def update_scale(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/update_scale")

    pool_err = require_pool(pg_pool, "/update_scale", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["scale", "project_id", "plan_id", "page_number"], "/update_scale", rid)
    if not valid:
        return err

    scale = params["scale"]
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    page_number = int(params["page_number"])

    await pg_execute(
        pg_pool,
        "UPDATE models SET scale = $1 WHERE LOWER(project_id) = LOWER($2) AND LOWER(plan_id) = LOWER($3) AND page_number = $4",
        [scale, project_id, plan_id, page_number],
        query_name="update_scale"
    )
    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/update_scale",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(status="success"))


@app.post("/load_scale")
async def load_scale(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_scale")

    pool_err = require_pool(pg_pool, "/load_scale", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number"], "/load_scale", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    page_number = int(params["page_number"])

    row = await pg_fetch_one(
        pg_pool,
        "SELECT scale FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, page_number],
        query_name="load_scale"
    )
    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_scale",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(scale=row["scale"] if row else None))


@app.post("/floorplan_to_3d")
async def floorplan_to_3d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/floorplan_to_3d")

    pool_err = require_pool(pg_pool, "/floorplan_to_3d", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id", "page_number"], "/floorplan_to_3d", rid)
    if not valid:
        return err

    walls_2d_JSON = params.get("walls_2d")
    polygons_JSON = params.get("polygons")
    project_id = params["project_id"]
    user_id = params["user_id"]
    plan_id = params["plan_id"]
    scale = params.get("scale")
    index = params["page_number"]
    index_int = int(index)

    model_2d_path = "/tmp/walls_2d.json"
    with open(model_2d_path, 'w') as f:
        json.dump(walls_2d_JSON, f)
    polygons_path = "/tmp/polygons.json"
    with open(polygons_path, 'w') as f:
        json.dump(polygons_JSON, f)

    if not scale:
        row = await pg_fetch_one(
            pg_pool,
            "SELECT scale FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
            [project_id, plan_id, index_int],
            query_name="floorplan_to_3d__get_scale"
        )
        scale = row["scale"] if row else None

    async with timed_step("extrapolate_3d", request_id=rid):
        # Use cached HYPERPARAMETERS instead of reading YAML per-request
        floor_plan_modeller_3d = Extrapolate3D(HYPERPARAMETERS)
        walls_3d, polygons_3d, walls_3d_path, polygons_3d_path = floor_plan_modeller_3d.extrapolate(scale, model_2d_path=model_2d_path, polygons_path=polygons_path)
        walls_3d, polygons_3d = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d, polygons_3d)
        gltf_paths = floor_plan_modeller_3d.gltf(model_2d_path=model_2d_path, polygons_path=polygons_path)
        model_3d_path = floor_plan_modeller_3d.save_plot_3d(walls_3d_path, polygons_3d_path)

    row = await pg_fetch_one(
        pg_pool,
        "SELECT model_2d->'metadata' AS metadata FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, index_int],
        query_name="floorplan_to_3d__get_metadata"
    )
    metadata = parse_jsonb(row["metadata"]) if row else None

    async with timed_step("upload_3d_assets", request_id=rid):
        upload_floorplan(model_3d_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
        for gltf_path in gltf_paths:
            upload_floorplan(gltf_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2), directory="gltf")

    await insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/floorplan_to_3d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(walls_3d=walls_3d, polygons=polygons_3d, metadata=metadata))


@app.post("/update_floorplan_to_3d")
async def update_floorplan_to_3d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/update_floorplan_to_3d")

    pool_err = require_pool(pg_pool, "/update_floorplan_to_3d", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id", "page_number"], "/update_floorplan_to_3d", rid)
    if not valid:
        return err

    walls_3d = params.get("walls_3d")
    polygons_3d = params.get("polygons")
    project_id = params["project_id"]
    user_id = params["user_id"]
    plan_id = params["plan_id"]
    scale = params.get("scale")
    index = params["page_number"]

    await insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    await insert_model_3d_revision(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/update_floorplan_to_3d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(status="success"))


@app.post("/load_3d_all")
async def load_3d_all(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_3d_all")

    pool_err = require_pool(pg_pool, "/load_3d_all", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id"], "/load_3d_all", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    plan_id = params["plan_id"]

    rows = await pg_fetch_all(
        pg_pool,
        "SELECT page_number, scale, model_3d, model_2d->'metadata' AS metadata FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        [project_id, plan_id],
        query_name="load_3d_all"
    )

    walls_3d_all = dict(pages=list())
    for row in rows:
        if not row["model_3d"]:
            continue
        model_3d = parse_jsonb(row["model_3d"])
        if not model_3d:
            continue
        metadata = parse_jsonb(row["metadata"]) or {}
        page = dict(
            plan_id=plan_id,
            page_number=row["page_number"],
            walls_3d=model_3d.get("walls_3d", []),
            polygons=model_3d.get("polygons", []),
            scale=row["scale"],
            **metadata,
        )
        walls_3d_all["pages"].append(page)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_3d_all",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             page_count=len(walls_3d_all["pages"]))
    return respond_with_UI_payload(walls_3d_all)


@app.post("/load_3d_revision")
async def load_3d_revision(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_3d_revision")

    pool_err = require_pool(pg_pool, "/load_3d_revision", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number", "revision_number"], "/load_3d_revision", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    page_number = int(params["page_number"])
    revision_number = int(params["revision_number"])

    row = await pg_fetch_one(
        pg_pool,
        "SELECT model FROM model_revisions_3d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3 AND revision_number = $4",
        [project_id, plan_id, page_number, revision_number],
        query_name="load_3d_revision"
    )
    walls_3d_JSON = parse_jsonb(row["model"]) if row else dict()

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_3d_revision",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(walls_3d_JSON)


@app.post("/load_available_revision_numbers_3d")
async def load_available_revision_numbers_3d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_available_revision_numbers_3d")

    pool_err = require_pool(pg_pool, "/load_available_revision_numbers_3d", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number"], "/load_available_revision_numbers_3d", rid)
    if not valid:
        return err
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    page_number = int(params["page_number"])

    rows = await pg_fetch_all(
        pg_pool,
        "SELECT revision_number FROM model_revisions_3d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, page_number],
        query_name="load_available_revision_numbers_3d"
    )
    revision_numbers = [row["revision_number"] for row in rows if row["revision_number"] is not None]

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_available_revision_numbers_3d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             revision_count=len(revision_numbers))
    return respond_with_UI_payload(revision_numbers)


@app.post("/generate_drywall_overlaid_floorplan_download_signed_URL")
async def generate_drywall_overlaid_floorplan_download_signed_URL(request: Request) -> str:
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/generate_drywall_overlaid_floorplan_download_signed_URL")

    pool_err = require_pool(pg_pool, "/generate_drywall_overlaid_floorplan_download_signed_URL", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number"], "/generate_drywall_overlaid_floorplan_download_signed_URL", rid)
    if not valid:
        return err

    index = int(params["page_number"])
    project_id = params["project_id"]
    plan_id = params["plan_id"]

    # Wait for plan to complete
    plan_row = await pg_fetch_one(
        pg_pool,
        "SELECT pages, status FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        [project_id, plan_id],
        query_name="gen_drywall_url__get_plan"
    )
    if not plan_row:
        return respond_with_UI_payload(dict(error="Floor Plan does not exist"))

    n_pages = plan_row["pages"]
    status = plan_row["status"]
    timeout = from_unix_epoch() + (n_pages * 120)
    while status != "COMPLETED" and from_unix_epoch() < timeout:
        await asyncio.sleep(2)
        status_row = await pg_fetch_one(
            pg_pool,
            "SELECT status FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
            [project_id, plan_id],
            query_name="gen_drywall_url__poll_status"
        )
        if not status_row:
            return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
        status = status_row["status"]

    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed within timeout"), status_code=500)

    row = await pg_fetch_one(
        pg_pool,
        "SELECT target_drywalls FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, index],
        query_name="gen_drywall_url__get_target_drywalls"
    )
    if not row or not row["target_drywalls"]:
        return respond_with_UI_payload(dict(error="Drywall overlay not found for this page"), status_code=404)

    drywall_overlaid_floorplan_source_path = row["target_drywalls"]
    _, _, _, blob_path = drywall_overlaid_floorplan_source_path.split('/', 3)

    client = get_gcs_client()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="GET",
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/generate_drywall_overlaid_floorplan_download_signed_URL",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return url


@app.post("/remove_floorplan")
async def remove_floorplan(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/remove_floorplan")

    pool_err = require_pool(pg_pool, "/remove_floorplan", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id"], "/remove_floorplan", rid)
    if not valid:
        return err

    project_id = params["project_id"]
    user_id = params["user_id"]
    plan_id = params["plan_id"]

    await delete_floorplan(project_id, plan_id, user_id, pg_pool, CREDENTIALS)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/remove_floorplan",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(status="success"))


@app.post("/compute_takeoff")
async def compute_takeoff(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/compute_takeoff")

    pool_err = require_pool(pg_pool, "/compute_takeoff", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "plan_id", "user_id", "page_number"], "/compute_takeoff", rid)
    if not valid:
        return err

    walls_3d_JSON = params.get("walls_3d", list())
    polygons_JSON = params.get("polygons", list())
    index = params["page_number"]
    index_int = int(index)
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    user_id = params["user_id"]
    revision_number = params.get("revision_number", '')

    row = await pg_fetch_one(
        pg_pool,
        "SELECT scale FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, index_int],
        query_name="compute_takeoff__get_scale"
    )
    scale = row["scale"] if row else None

    async with timed_step("download_floorplan", request_id=rid, plan_id=plan_id):
        pdf_path = Path("/tmp/floor_plan.PDF")
        download_floorplan(plan_id, project_id, CREDENTIALS, destination_path=pdf_path)

    if not walls_3d_JSON:
        if revision_number:
            rev_row = await pg_fetch_one(
                pg_pool,
                "SELECT model FROM model_revisions_3d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3 AND revision_number = $4",
                [project_id, plan_id, index_int, int(revision_number)],
                query_name="compute_takeoff__get_revision_model"
            )
            walls_3d_JSON = parse_jsonb(rev_row["model"]) if rev_row and rev_row["model"] else list()
        else:
            model_row = await pg_fetch_one(
                pg_pool,
                "SELECT model_3d FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
                [project_id, plan_id, index_int],
                query_name="compute_takeoff__get_model_3d"
            )
            walls_3d_JSON = parse_jsonb(model_row["model_3d"]) if model_row and model_row["model_3d"] else list()

        if walls_3d_JSON is None:
            walls_3d_JSON = list()

    async with timed_step("compute_takeoff_calculation", request_id=rid):
        # Use cached HYPERPARAMETERS
        floor_plan_modeller_3d = Extrapolate3D(HYPERPARAMETERS)
        if scale != "1/4``=1`0``":
            pixel_aspect_ratio_new = floor_plan_modeller_3d.compute_pixel_aspect_ratio(scale, HYPERPARAMETERS["pixel_aspect_ratio_to_feet"])
            walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.recompute_dimensions_walls_and_polygons(walls_3d_JSON, polygons_JSON, pixel_aspect_ratio_new, pdf_path)
        walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d_JSON, polygons_JSON)

        drywall_takeoff = dict(total=dict(roof=0, wall=0), per_drywall=dict(roof=defaultdict(lambda: 0), wall=defaultdict(lambda: 0)))
        for wall in walls_3d_JSON:
            surface_area = wall["height"] * wall["length"]
            drywall_count = 0
            for drywall in wall["surfaces_drywall"]:
                if drywall["enabled"]:
                    waste_factor = drywall["waste_factor"]
                    if isinstance(waste_factor, float):
                        waste_factor = float(waste_factor)
                    elif isinstance(waste_factor, str) and waste_factor.find('%') != -1:
                        if waste_factor.find('-') != -1:
                            waste_factor = float(waste_factor.strip('%').split('-')[1]) / 100
                        else:
                            waste_factor = float(waste_factor.strip('%')) / 100
                    else:
                        try:
                            waste_factor = float(waste_factor)
                        except (ValueError, TypeError):
                            waste_factor = 0
                    drywall_takeoff["per_drywall"]["wall"][drywall["type"]] += surface_area * (1 + waste_factor)
                    drywall_count += 1
            drywall_takeoff["total"]["wall"] += drywall_count * surface_area

        for polygon in polygons_JSON:
            surface_area = floor_plan_modeller_3d.compute_updated_area_polygon(
                polygon["vertices"], polygon["area"], polygon["slope"], polygon["tilt_axis"]
            )
            waste_factor = polygon["surface_drywall"]["waste_factor"]
            if isinstance(waste_factor, float):
                waste_factor = float(waste_factor)
            elif isinstance(waste_factor, str) and waste_factor.find('%') != -1:
                if waste_factor.find('-') != -1:
                    waste_factor = float(waste_factor.strip('%').split('-')[1]) / 100
                else:
                    waste_factor = float(waste_factor.strip('%')) / 100
            else:
                try:
                    waste_factor = float(waste_factor)
                except (ValueError, TypeError):
                    waste_factor = 0
            drywall_takeoff["per_drywall"]["roof"][polygon["surface_drywall"]["type"]] += surface_area * (1 + waste_factor)
            drywall_takeoff["total"]["roof"] += surface_area

        drywall_takeoff["total"]["wall"] = round(drywall_takeoff["total"]["wall"], 2)
        drywall_takeoff["total"]["roof"] = round(drywall_takeoff["total"]["roof"], 2)
        for key in drywall_takeoff["per_drywall"]["wall"]:
            drywall_takeoff["per_drywall"]["wall"][key] = round(drywall_takeoff["per_drywall"]["wall"][key], 2)
        for key in drywall_takeoff["per_drywall"]["roof"]:
            drywall_takeoff["per_drywall"]["roof"][key] = round(drywall_takeoff["per_drywall"]["roof"][key], 2)

    await insert_takeoff(drywall_takeoff, index, plan_id, user_id, project_id, revision_number if revision_number else None, pg_pool, CREDENTIALS)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/compute_takeoff",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(drywall_takeoff)


@app.get("/insert_templates")
async def insert_templates():
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/insert_templates")

    pool_err = require_pool(pg_pool, "/insert_templates", rid)
    if pool_err:
        return pool_err

    # Lazy import — pandas/openpyxl only needed here, not at startup
    import pandas as pd

    def parse_fire_rating(description: str):
        if "TYPE C" in description:
            return "Type C"
        if "TYPE X" in description:
            return "Type X"
        return None

    def parse_lightweight(description: str):
        return "LITE" in description.upper()

    def parse_wide_stretch(description: str):
        return "WIDE-STRETCH" in description.upper()

    def parse_thickness(description: str):
        match = re.search(r'(\d+\/\d+)"', description)
        if match:
            fraction = match.group(1)
            numerator, denominator = fraction.split("/")
            return float(numerator) / float(denominator)
        return None

    def generate_random_colors(n, seed=0):
        rng = np.random.default_rng(seed)
        total_colors = 256**3
        excluded_index = 255 * 256 * 256 + 0 * 256 + 0
        indices = rng.choice(total_colors - 1, size=n, replace=False)
        indices = np.where(indices >= excluded_index, indices + 1, indices)
        colors = list()
        for index in indices:
            r = index // (256 * 256)
            g = (index // 256) % 256
            b = index % 256
            colors.append((int(r), int(g), int(b)))
        return [dict(r=int(color[0]), g=int(color[1]), b=int(color[2])) for color in colors]

    dataframe = pd.read_excel("Drywall_P_Code_20260122.xlsx")
    product_color_codes = generate_random_colors(dataframe.size)
    insert_count = 0

    for (_, row_data), product_color_code in zip(dataframe.iterrows(), product_color_codes):
        if pd.isna(row_data["user10"]) or pd.isna(row_data["user11"]) or pd.isna(row_data["PRODUCT_CAT_CODE"]) or pd.isna(row_data["PRODUCT_CAT_DESC"]) or not isinstance(row_data["PRODUCT_CAT_CODE"], int):
            continue
        sku_description = str(row_data["user11"]).upper()
        color_code_json = json.dumps(product_color_code)

        await pg_execute(
            pg_pool,
            """
            INSERT INTO sku (
                sku_id, sku_description, product_cat_code, product_cat_description,
                thickness_inches, fire_rating, is_lightweight, is_wide_stretch, color_code
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
            ON CONFLICT (sku_id) DO NOTHING
            """,
            [
                str(row_data["user10"]), str(row_data["user11"]),
                int(row_data["PRODUCT_CAT_CODE"]), str(row_data["PRODUCT_CAT_DESC"]),
                parse_thickness(sku_description), parse_fire_rating(sku_description),
                parse_lightweight(sku_description), parse_wide_stretch(sku_description),
                color_code_json
            ],
            query_name="insert_templates"
        )
        insert_count += 1

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/insert_templates",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             templates_inserted=insert_count)
    return respond_with_UI_payload(dict(status="success", templates_inserted=insert_count))
