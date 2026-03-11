import json
import hashlib
import sys
import os
import time
import re  # Added for bulletproof JSON extraction
from pathlib import Path
from contextlib import asynccontextmanager

import asyncpg
import cv2
from time import sleep
from random import uniform
from ruamel.yaml import YAML

from google.cloud.storage import Client as CloudStorageClient
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, DeadlineExceeded

from prompt import ARCHITECTURAL_DRAWING_CLASSIFIER, ArchitecturalDrawingClassifierResponse


# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------

def log_json(severity: str, message: str, **kwargs):
    """Emit a single structured JSON log line to stdout (Cloud Logging compatible)."""
    payload = {"severity": severity, "message": message}
    payload.update(kwargs)
    print(json.dumps(payload, default=str), flush=True)


@asynccontextmanager
async def timed_step(step_name: str, request_id: str = "", **extra):
    """Async context manager that logs step duration on exit."""
    start = time.perf_counter()
    log_json("INFO", "STEP_START", step=step_name, request_id=request_id, **extra)
    error_msg = None
    try:
        yield
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if error_msg:
            log_json("ERROR", "STEP_FAILED", step=step_name, request_id=request_id,
                     duration_ms=duration_ms, error=error_msg, **extra)
        else:
            log_json("INFO", "STEP_COMPLETE", step=step_name, request_id=request_id,
                     duration_ms=duration_ms, **extra)


# ---------------------------------------------------------------------------
# Configuration Loaders (called once at startup, cached)
# ---------------------------------------------------------------------------

def load_gcp_credentials() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("gcp.yaml", 'r') as f:
        credentials = yaml.load(f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials["service_drywall_account_key"]
    return credentials


def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("hyperparameters.yaml", 'r') as f:
        hyperparameters = yaml.load(f)
    return hyperparameters


# ---------------------------------------------------------------------------
# GCS Client (shared singleton — created once, reused everywhere)
# ---------------------------------------------------------------------------

_gcs_client = None

def get_gcs_client() -> CloudStorageClient:
    """Return a shared GCS client. Created once on first call."""
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = CloudStorageClient()
        log_json("INFO", "GCS_CLIENT_CREATED")
    return _gcs_client


# ---------------------------------------------------------------------------
# PostgreSQL Connection Pool
# ---------------------------------------------------------------------------

async def create_pg_pool(credentials) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool."""
    pg_config = credentials["PostgreSQL"]
    try:
        pool = await asyncpg.create_pool(
            host=pg_config["host"],
            port=pg_config["port"],
            database=pg_config["database"],
            user=pg_config["user"],
            password=pg_config["password"],
            min_size=pg_config.get("min_pool_size", 2),
            max_size=pg_config.get("max_pool_size", 10),
            command_timeout=60,
        )
        log_json("INFO", "PG_POOL_CREATED", host=pg_config["host"],
                 database=pg_config["database"],
                 min_size=pg_config.get("min_pool_size", 2),
                 max_size=pg_config.get("max_pool_size", 10))
        return pool
    except Exception as exc:
        log_json("ERROR", "PG_POOL_FAILED", host=pg_config["host"],
                 error=f"{type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core DB Helpers (with built-in observability)
# ---------------------------------------------------------------------------

async def pg_fetch_all(pool: asyncpg.Pool, query: str, params: list = None,
                       query_name: str = "unnamed") -> list:
    """Execute a SELECT and return all rows as list of Record objects."""
    start = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *(params or []))
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("INFO", "DB_FETCH_ALL", query=query_name,
                 duration_ms=duration_ms, row_count=len(rows))
        return rows
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("ERROR", "DB_FETCH_ALL_FAILED", query=query_name,
                 duration_ms=duration_ms, error=f"{type(exc).__name__}: {exc}")
        raise


async def pg_fetch_one(pool: asyncpg.Pool, query: str, params: list = None,
                       query_name: str = "unnamed"):
    """Execute a SELECT and return first row (or None)."""
    start = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *(params or []))
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("INFO", "DB_FETCH_ONE", query=query_name,
                 duration_ms=duration_ms, found=row is not None)
        return row
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("ERROR", "DB_FETCH_ONE_FAILED", query=query_name,
                 duration_ms=duration_ms, error=f"{type(exc).__name__}: {exc}")
        raise


async def pg_execute(pool: asyncpg.Pool, query: str, params: list = None,
                     query_name: str = "unnamed") -> str:
    """Execute an INSERT/UPDATE/DELETE and return status string."""
    start = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            status = await conn.execute(query, *(params or []))
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("INFO", "DB_EXECUTE", query=query_name,
                 duration_ms=duration_ms, status=status)
        return status
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("ERROR", "DB_EXECUTE_FAILED", query=query_name,
                 duration_ms=duration_ms, error=f"{type(exc).__name__}: {exc}")
        raise


# ---------------------------------------------------------------------------
# JSONB Helper
# ---------------------------------------------------------------------------

def parse_jsonb(value):
    """Safely parse a JSONB value from asyncpg (could be str, dict, or None)."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value  # already a dict/list from asyncpg


# ---------------------------------------------------------------------------
# Database Operations (migrated from BigQuery)
# ---------------------------------------------------------------------------

async def insert_model_2d(
    model_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    GCS_URL_floorplan_page,
    GCS_URL_target_drywalls_page,
    pool,
    credentials
):
    """Upsert a 2D model into the models table.
    BQ MERGE → PG INSERT ... ON CONFLICT using index name.
    """
    page_number = int(page_number)

    if not model_2d.get("metadata", None):
        row = await pg_fetch_one(
            pool,
            "SELECT model_2d->'metadata' AS metadata FROM models "
            "WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
            [project_id, plan_id, page_number],
            query_name="insert_model_2d__fetch_metadata"
        )
        if row:
            metadata = parse_jsonb(row["metadata"])
            if metadata:
                model_2d["metadata"] = metadata

    model_2d_json = json.dumps(model_2d)
    source = GCS_URL_floorplan_page or ''
    target_drywalls = GCS_URL_target_drywalls_page or ''
    scale = scale or ''

    await pg_execute(
        pool,
        """
        INSERT INTO models (
            plan_id, project_id, user_id, page_number, scale,
            model_2d, model_3d, takeoff, source, target_drywalls,
            created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6::jsonb, '{}'::jsonb, '{}'::jsonb, $7, $8,
            NOW(), NOW()
        )
        # ON CONFLICT ON CONSTRAINT idx_models_project_plan_page
        ON CONFLICT (LOWER(project_id), LOWER(plan_id), page_number)
        DO UPDATE SET
            model_2d = EXCLUDED.model_2d,
            scale = CASE WHEN EXCLUDED.scale = '' THEN models.scale ELSE EXCLUDED.scale END,
            user_id = EXCLUDED.user_id,
            updated_at = NOW()
        """,
        [plan_id, project_id, user_id, page_number, scale,
         model_2d_json, source, target_drywalls],
        query_name="insert_model_2d"
    )


async def is_duplicate(pool, credentials, pdf_path, project_id):
    """Check if a PDF with the same sha256 already exists for this project."""
    sha_256 = sha256(pdf_path)
    rows = await pg_fetch_all(
        pool,
        "SELECT plan_id, sha256, status FROM plans WHERE LOWER(project_id) = LOWER($1)",
        [project_id],
        query_name="is_duplicate"
    )
    for row in rows:
        if row["sha256"] == sha_256:
            if row["status"] == "FAILED":
                await delete_plan(pool, credentials, row["plan_id"], project_id)
                return False
            return row["plan_id"]
    return False


async def delete_plan(pool, credentials, plan_id, project_id):
    """Delete a plan row by project_id + plan_id."""
    await pg_execute(
        pool,
        "DELETE FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        [project_id, plan_id],
        query_name="delete_plan"
    )


# ---------------------------------------------------------------------------
# Non-DB Helpers (unchanged logic, no BQ dependency)
# ---------------------------------------------------------------------------

def sha256(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_floorplan(plan_path, plan_id, project_id, credentials, index=None, directory=None):
    client = get_gcs_client()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if directory:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{directory}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{directory}/{blob_object_name}"
    else:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{blob_object_name}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(plan_path)
    return f"gs://{credentials['CloudStorage']['bucket_name']}/{blob_path}"


def load_floorplan_to_structured_2d_ID_token(credentials):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["floorplan_to_structured_2d"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token
    return id_token


def load_vertex_ai_client(credentials, region="us-central1"):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = GenerativeModel(credentials["VertexAI"]["llm"]["model_name"])
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config


def classify_plan(plan_path, vertex_ai_client_parameters):
    vertex_ai_client, vertex_ai_generation_config, vertex_ai_max_retry = vertex_ai_client_parameters
    plan_BGR = cv2.imread(plan_path)
    if plan_BGR is None:
        log_json("ERROR", "CLASSIFY_PLAN_FAILED", error=f"Could not read image: {plan_path}")
        return {"plan_type": "UNKNOWN", "confidence": 0.0}
    _, canvas_buffer_array = cv2.imencode(".png", plan_BGR)
    bytes_canvas = canvas_buffer_array.tobytes()
    system = Content(role="model", parts=[Part.from_text(ARCHITECTURAL_DRAWING_CLASSIFIER)])
    query = Content(role="user", parts=[
        Part.from_data(data=bytes_canvas, mime_type="image/png"),
        Part.from_text("Classify this architectural drawing.")
    ])

    for attempt in range(vertex_ai_max_retry):
        try:
            response = vertex_ai_client.generate_content(
                [system, query],
                generation_config=vertex_ai_generation_config,
            )
            response_text = response.text.strip()
            
            # --- BULLETPROOF JSON EXTRACTION ---
            # Finds the exact JSON block and ignores all conversational text or markdown
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON object found in response: {response_text}")
                
            clean_json_string = json_match.group(0)
            classification = json.loads(clean_json_string)
            
            validated = ArchitecturalDrawingClassifierResponse(**classification)
            return validated.model_dump()
        except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded) as e:
            log_json("WARNING", "CLASSIFY_PLAN_RETRY", attempt=attempt + 1,
                     max_retry=vertex_ai_max_retry, error=str(e))
            sleep(uniform(1, 3))
        except Exception as e:
            log_json("WARNING", "CLASSIFY_PLAN_RETRY", attempt=attempt + 1,
                     max_retry=vertex_ai_max_retry, error=str(e))
            sleep(uniform(1, 3))

    log_json("ERROR", "CLASSIFY_PLAN_EXHAUSTED", plan_path=str(plan_path),
             max_retry=vertex_ai_max_retry)
    return {"plan_type": "UNKNOWN", "confidence": 0.0}
