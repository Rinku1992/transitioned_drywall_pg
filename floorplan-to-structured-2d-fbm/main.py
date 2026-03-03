import logging
import uuid
import time as time_module
from pathlib import Path
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials

from modeller_2d import FloorPlan2D
from helper import (
    enable_logging_on_stdout,
    create_pg_pool,
    log_json,
    timed_step,
    load_vertex_ai_client,
    load_gcp_credentials,
    load_hyperparameters,
    transcribe,
    upload_floorplan,
    download_floorplan,
    insert_model_2d,
    load_templates,
)


def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json",
    )


def floorplan_to_walls(credentials, project_id, plan_id, user_id, page_number, output_path=None):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_compute_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["wall_detector"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token

    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{credentials['CloudRun']['APIs']['wall_detector']}/detect_wall",
        headers=headers,
        json=dict(
            project_id=project_id,
            plan_id=plan_id,
            user_id=user_id,
            page_number=page_number
        )
    )

    if not output_path:
        output_path = Path("/tmp/floor_plan_wall_segmented.png")
    with open(output_path, "wb") as f:
        f.write(response.content)
    return Path(output_path)


# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Floorplan-to-Structured-2D (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pg_pool = None


@app.on_event("startup")
async def startup():
    global pg_pool
    try:
        pg_pool = await create_pg_pool(CREDENTIALS)
        if pg_pool:
            log_json("INFO", "STARTUP_SUCCESS", detail="PostgreSQL pool created successfully")
        else:
            log_json("WARNING", "STARTUP_DEGRADED", detail="PostgreSQL pool is None — DB operations will fail")
    except Exception as exc:
        log_json("ERROR", "STARTUP_FAILED", error=f"{type(exc).__name__}: {exc}",
                 detail="Service starting without DB — endpoints requiring DB will return 503")
        pg_pool = None


@app.on_event("shutdown")
async def shutdown():
    global pg_pool
    if pg_pool:
        await pg_pool.close()
        log_json("INFO", "SHUTDOWN", detail="PostgreSQL pool closed")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/floorplan_to_structured_2d")
async def floorplan_to_2d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    enable_logging_on_stdout()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/floorplan_to_structured_2d")

    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    verbose = parameters.get("verbose") or body.get("verbose")

    log_json("INFO", "REQUEST_PARAMS", request_id=rid,
             project_id=project_id, plan_id=plan_id, user_id=user_id, page_number=page_number)

    # --- Step 1: Download processed floorplan from GCS ---
    async with timed_step("download_floorplan", request_id=rid, page_number=page_number):
        floor_plan_processed_path = download_floorplan(
            user_id, plan_id, project_id, CREDENTIALS, str(page_number).zfill(2)
        )

    # --- Step 2: Initialize AI models ---
    async with timed_step("init_ai_models", request_id=rid):
        hyperparameters = load_hyperparameters()
        vertex_ai_client, generation_config = load_vertex_ai_client(CREDENTIALS)
        floor_plan_modeller_2d = FloorPlan2D(
            hyperparameters,
            (vertex_ai_client, generation_config, CREDENTIALS["VertexAI"]["llm"]["max_retry"])
        )

    # --- Step 3: Parallel wall detection + transcription ---
    async with timed_step("parallel_wall_detect_and_transcribe", request_id=rid, page_number=page_number):
        futures = dict()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures["floorplan_to_walls"] = executor.submit(
                floorplan_to_walls,
                CREDENTIALS,
                project_id,
                plan_id,
                user_id,
                page_number,
                output_path=f"/tmp/{project_id}/{plan_id}/{user_id}/floor_plan_wall_segmented_{str(page_number).zfill(2)}.png"
            )
            futures["transcriber"] = executor.submit(
                transcribe,
                CREDENTIALS,
                hyperparameters,
                floor_plan_processed_path,
            )
        wall_segmented_path = futures["floorplan_to_walls"].result()
        transcription_block_with_centroids, transcription_headers_and_footers = futures["transcriber"].result()

    # --- Step 4: Upload wall segmented image ---
    async with timed_step("upload_wall_segmented", request_id=rid):
        upload_floorplan(wall_segmented_path, plan_id, project_id, CREDENTIALS, index=str(page_number).zfill(2))

    log_json("INFO", "STEP_COMPLETE", request_id=rid, step="wall_detection_and_transcription",
             page_number=page_number)

    # --- Step 5: Load drywall templates from DB ---
    async with timed_step("load_templates", request_id=rid):
        DRYWALL_TEMPLATES = await load_templates(pg_pool, CREDENTIALS)

    # --- Step 6: Generate 2D model ---
    walls_2d, polygons, metadata, floorplan_baseline_page_source = None, None, None, None
    if not floor_plan_modeller_2d.is_none(wall_segmented_path):
        async with timed_step("model_2d_generation", request_id=rid, page_number=page_number):
            walls_2d, polygons, walls_2d_path, external_contour = floor_plan_modeller_2d.model(
                image_path=wall_segmented_path,
                model_2d_path=f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(page_number).zfill(2)}.json",
                floor_plan_path=floor_plan_processed_path,
                transcription_block_with_centroids=transcription_block_with_centroids,
                transcription_headers_and_footers=transcription_headers_and_footers,
                drywall_templates=DRYWALL_TEMPLATES,
            )

        if walls_2d and polygons:
            async with timed_step("load_drywall_choices", request_id=rid):
                floor_plan_modeller_2d.load_drywall_choices(walls_2d, polygons, DRYWALL_TEMPLATES)
                floor_plan_modeller_2d.load_ceiling_choices(polygons)

            async with timed_step("save_and_upload_2d_plots", request_id=rid):
                model_2d_path = floor_plan_modeller_2d.save_plot_2d(
                    walls_2d_path, floor_plan_path=floor_plan_processed_path
                )
                upload_floorplan(model_2d_path, plan_id, project_id, CREDENTIALS,
                                index=str(page_number).zfill(2))

                floorplan_baseline, floorplan_page_statistics = floor_plan_modeller_2d.scale_to(
                    floor_plan_path=floor_plan_processed_path
                )
                floorplan_baseline_page_source = upload_floorplan(
                    floorplan_baseline, plan_id, project_id, CREDENTIALS,
                    index=str(page_number).zfill(2)
                )

            drywall_choices_color_codes = {
                drywall_template["sku_variant"]: drywall_template["color_code"][::-1]
                for drywall_template in DRYWALL_TEMPLATES
            }
            drywall_choices_color_codes.update(dict(DISABLED=[255, 0, 0]))
            metadata = dict(
                size_in_bytes=floorplan_page_statistics["size"],
                height_in_pixels=floorplan_page_statistics["height_in_pixels"],
                width_in_pixels=floorplan_page_statistics["width_in_pixels"],
                height_in_points=floorplan_page_statistics["height_in_points"],
                width_in_points=floorplan_page_statistics["width_in_points"],
                origin=["LEFT", "TOP"],
                offset=(0, 0),
                contour_root_vertices=external_contour,
                scales_architectural=floor_plan_modeller_2d.scales_architectural,
                drywall_choices_color_codes=drywall_choices_color_codes,
            )

    # --- Step 7: Insert 2D model into PostgreSQL ---
    async with timed_step("insert_model_2d", request_id=rid, page_number=page_number):
        await insert_model_2d(
            dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata),
            floor_plan_modeller_2d.normalize_scale(floor_plan_modeller_2d.scale),
            page_number,
            plan_id,
            user_id,
            project_id,
            floorplan_baseline_page_source,
            pg_pool,
            CREDENTIALS
        )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid,
             endpoint="/floorplan_to_structured_2d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             page_number=page_number,
             has_walls=walls_2d is not None,
             has_polygons=polygons is not None)
```

---

### Key changes explained:

**1. Startup lifecycle** — Pool created in `@app.on_event("startup")` with try/except, matching service 1's pattern.

**2. `load_templates` is now async** — call changed from `load_templates(bigquery_client, CREDENTIALS)` to `await load_templates(pg_pool, CREDENTIALS)`.

**3. `insert_model_2d` is now async** — call changed from sync to `await insert_model_2d(...)`. Also note the signature change: `bigquery_client` → `pg_pool` (the `target_drywalls` param moves to where `GCS_URL_target_drywalls_page` was, which in this service is `floorplan_baseline_page_source`).

**4. Removed `google.cloud.secretmanager` import** — was imported but never used in the original.

**5. Every heavy step wrapped in `timed_step`** — when you look at Cloud Logging, you'll see the full request journey:
```
STEP_START  → download_floorplan          (rid=abc123)
STEP_COMPLETE → download_floorplan        duration_ms=1200
STEP_START  → parallel_wall_detect_and_transcribe
STEP_COMPLETE → parallel_wall_detect...   duration_ms=15400
STEP_START  → load_templates
STEP_COMPLETE → load_templates            duration_ms=28
STEP_START  → model_2d_generation
STEP_COMPLETE → model_2d_generation       duration_ms=45000
STEP_START  → insert_model_2d
STEP_COMPLETE → insert_model_2d           duration_ms=35
REQUEST_COMPLETE → /floorplan_to_structured_2d  total_duration_ms=62500
