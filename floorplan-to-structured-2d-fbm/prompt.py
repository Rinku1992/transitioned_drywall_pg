from typing import List, Union, Optional, Tuple, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import math


WALL_RECTIFIER = """
  You are a senior architectural plan-correction specialist with 20+ years of experience in residential and commercial floor plans.

  You do NOT trust automated detections blindly.
  You treat detected walls and drywalls as noisy suggestions.

  Your responsibility is to:
    - Correct wall alignment errors.
    - Extend or shorten or shift walls to meet logical intersections.
    - Add missing walls where enclosure logic requires them.
    - Remove false-positive wall fragments.
    - Distinguish structural walls vs drywalls and correct the drywall positioning 

  PROVIDED:
    1. A polygon represented by a list of vertices and the polygon perimeter lines/edges joining the vertices with origin set to LEFT, TOP of the original floorplan and offset set to (0, 0):
      Vertices: [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
      Perimeter wall endpoints: [
        wall: (X1, Y1) → (X2, Y2),
        wall: (X2, Y2) → (X3, Y3),
        wall: (X4, Y4) → (X3, Y3),
        wall: (X1, Y1) → (X4, Y4)
      ]

    2. A cropped snapshot of the room or the polygon from Architectural Drawing in png format inscribed with textual annotations containing the name of the room it belongs to with the wall line dimensions along with the following highlights,
      - The target polygon/room highlighted with transparent red color that corresponds with the provided polygon vertices computed from the whole floor plan using original offset on a different coordinate space but with same resolution and the area is inscribed with the room name information.
      - The target polygon/room's perimeter lines highlighted with blue bounding boxes that corresponds with provided polygon perimeter wall endpoints computed from the whole floor plan using original offset on a different coordinate space but with same resolution and the nearby regions are inscribed with textual annotations containing dimension marker and the dimension, width and height (optional) of the wall in `(feet) and ``(inches).

    3. Offset of the cropped snapshot,
      Offset: (X, Y)

  TASK:
    Analyze the architectural floor plan and highlighted wall segments accompanied by polygon vertices and it's perimeter wall endpoints to correct the floor plan following the `WALL_CORRECTION_INSTRUCTIONS` to minimize total wall discontinuity.

    WALL_CORRECTION_INSTRUCTIONS:
    - Focus only on perimeter walls surrounding the highlighted polygon in red color and discard any other walls. DO NOT invent walls that are far from the perimeter of the highlighted polygon.
    - Walls must form closed enclosures.
    - Wall endpoints within 3% of image width must be snapped together.
    - Wall endpoints are provided as a list of 4 integers with (X1, Y1) representing the beginning of the wall line and (X2, Y2) representing the end.
    - The perimeter wall is likely to be a horizontal one if, their `Y` coordinates are same or have very little difference in values but the difference between their 'X' coordinates have a greater value.
    - The perimeter wall is likely to be a vertical one if, their `X` coordinates are same or have very little difference in values but the difference between their 'Y' coordinates have a greater value.
    - Since the provided coordinates are computed on a different coordinate space having the whole floor plan, refer the provided offset (X, Y) of the provided cropped snapshot computed through comparing the provided coordinate integers with the relative position of the pixels in the provided snapshot.
    - The axis of the wall line on the floor plan should exactly align with the axis of the blue bounding box drawn on top.
    - Determine the shift in pixels needed (across X and Y) in case the blue blouding box is not perfectly aligned with the central axis of the wall line or is smaller / larger in length than the actual wall line.
    - Using the computed offset (X, Y) and the relative pixel position for the walls in provided snapshot, compute the corrected wall endpoints in the absolute coordinate space (Offset_X + relative_X_position_of_a_pixel, offset_Y + relative_Y_position_of_a_pixel).
    - Determine if walls may be shifted or resized to improve enclosure logic.
    - Missing walls must be inferred if a room boundary is incomplete.
    - When rules conflict, prioritize enclosure completeness over detected bounding box length.

  OUTPUT:
    Your output must be precise, code-aligned, and structured. You must reason spatially and geometrically. Do NOT describe the image. Do NOT repeat detected lines verbatim.
    **STRICTLY**
      - Do not generate additional content apart from the designated JSON.
      - You must output corrected wall geometry containing corrected list of all the perimeter wall endpoints of the highlighted polygon. The size of the list should ne greater than or equal to the provided list of perimeter wall endpoints since additional walls may only be added if required to form a closed polygon.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    [
      {{"X1": <corrected_X1_of_wall_perimeter_line_1>, "Y1": <corrected_Y1_of_wall_perimeter_line_1>, "X2": <corrected_X2_of_wall_perimeter_line_1>, "Y2": <corrected_Y2_of_wall_perimeter_line_1>}},
      {{"X1": <corrected_X1_of_wall_perimeter_line_2>, "Y1": <corrected_Y1_of_wall_perimeter_line_2>, "X2": <corrected_X2_of_wall_perimeter_line_2>, "Y2": <corrected_Y2_of_wall_perimeter_line_2>}},
      {{"X1": <corrected_X1_of_wall_perimeter_line_3>, "Y1": <corrected_Y1_of_wall_perimeter_line_3>, "X2": <corrected_X2_of_wall_perimeter_line_3>, "Y2": <corrected_Y2_of_wall_perimeter_line_3>}},
      {{"X1": <corrected_X1_of_wall_perimeter_line_4>, "Y1": <corrected_Y1_of_wall_perimeter_line_4>, "X2": <corrected_X2_of_wall_perimeter_line_4>, "Y2": <corrected_Y2_of_wall_perimeter_line_4>}},
      {{"X1": <corrected_X1_of_wall_perimeter_line_5>, "Y1": <corrected_Y1_of_wall_perimeter_line_5>, "X2": <corrected_X2_of_wall_perimeter_line_5>, "Y2": <corrected_Y2_of_wall_perimeter_line_5>}}
    ]
"""
DRYWALL_PREDICTOR_CALIFORNIA = """
California residential drywall estimator. Analyze the highlighted polygon and predict drywall specifications.

PROVIDED:
  1. Polygon vertices and perimeter wall endpoints with pre-computed dimension candidates
  2. Cropped floor plan image (red=target polygon, blue=perimeter walls, green=interior partitions)
  3. Nearby OCR transcription entries with centroids

EACH WALL includes:
  - wall: endpoint coordinates
  - dimension_candidates: pre-parsed dimensions sorted by confidence (high/medium/low)
    Use the highest-confidence candidate. If none, use pixel-measured fallback from image.

DRYWALL TEMPLATES: {drywall_templates}

CLASSIFICATION RULES:
  - Garage-adjacent / dwelling separation / corridor → 5/8" Type X, 1-hr rated (CBC R302, IRC R302.6)
  - Bathroom / laundry / kitchen wet wall → 1/2" MR or cement board
  - Standard interior (bedroom/living/hallway) → 1/2" regular gypsum
  - Ceiling → 1/2" regular (5/8" if joist span >16")
  - Use exact sku_variant and color_code from templates. Do not invent materials.
  - waste_factor: "8-12%" standard, "12-15%" complex geometry
  - layers: 1 unless code requires double layer

DIMENSION RULES:
  - Wall length: prefer dimension_candidates over pixel measurement. Confidence >=0.9 → trust directly.
  - Wall width: default 1 ft if unmarked
  - Wall height: default ceiling height if unmarked
  - Ceiling area: compute from polygon vertices (ignore slope for area calc)
  - All dimensions in feet

CEILING TYPE: Flat (default if ambiguous), Single-sloped, Gable, Tray, Barrel vault, Coffered, Combination, Soffit, Cove, Dome, Cloister Vault, Knee-Wall, Cathedral with Flat Center, Angled-Plane, Boxed-Beam
  - tilt_axis: "horizontal" | "vertical" | "NULL" (NULL if slope=0)
  - Positive slope if descending from origin, negative otherwise
  - Height = max height if sloped

ROOM NAME: text nearest polygon centroid, or NULL if not found.

WALL ORDER: output wall_parameters in same order as input perimeter walls. BLUE drywall before GREEN.

OUTPUT: JSON only, no additional text.
  {{
    "ceiling": {{
      "room_name": "<name / NULL>",
      "area": <sqft float>,
      "confidence": <0-1 float>,
      "ceiling_type": "<type code>",
      "height": <feet float>,
      "slope": <degrees float>,
      "slope_enabled": <bool>,
      "tilt_axis": "<horizontal/vertical/NULL>",
      "drywall_assembly": {{
        "material": "<sku_variant from template>",
        "color_code": [B, G, R],
        "thickness": <feet float>,
        "layers": <int>,
        "fire_rating": <hours float>,
        "waste_factor": "<percentage string>"
      }},
      "code_references": ["<ref1>", "<ref2>"],
      "recommendation": "<notes>"
    }},
    "wall_parameters": [
      {{
        "room_name": "<name / NULL>",
        "length": <feet float>,
        "confidence": <0-1 float>,
        "width": <feet float / null>,
        "height": <feet float>,
        "wall_type": "<type>",
        "drywall_assembly": {{
          "material": "<sku_variant>",
          "color_code": [B, G, R],
          "thickness": <feet float>,
          "layers": <int>,
          "fire_rating": <hours float>,
          "waste_factor": "<percentage string>"
        }},
        "code_references": ["<ref1>"],
        "recommendation": "<notes>"
      }}
    ]
  }}
"""

def ensure_not_nan(v: float) -> float:
    if v is None:
        return v
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise ValueError("NaN or Inf not allowed")
    return v

class DrywallAssembly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # material: str
    material: Union[str, dict]
    color_code: Tuple[int, int, int]
    thickness: float
    layers: int
    fire_rating: Optional[Union[str, float]]
    waste_factor: Union[str, int, float]

    @field_validator("thickness")
    @classmethod
    def validate_float(cls, v):
        return ensure_not_nan(v)

    @field_validator("color_code")
    @classmethod
    def validate_bgr(cls, v):
        if len(v) != 3:
            raise ValueError("color_code must be BGR tuple")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("Invalid BGR value")
        return v

class Ceiling(BaseModel):
    model_config = ConfigDict(extra="allow")
    room_name: Optional[str] = None
    area: Optional[Union[float, str]] = 0.0
    confidence: Optional[float] = Field(default=0.5, ge=0, le=1)
    ceiling_type: Optional[str] = ""
    height: Optional[Union[float, str]] = 0.0
    slope: Optional[Union[float, str]] = 0.0
    slope_enabled: Optional[bool] = False
    tilt_axis: Optional[Literal["horizontal", "vertical", "NULL"]] = None
    drywall_assembly: Optional[DrywallAssembly] = None
    code_references: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = None
    @field_validator("area", "height", "slope", mode="before")
    @classmethod
    def coerce_float(cls, v):
        if v is None:
            return 0.0
        try:
            return ensure_not_nan(float(v))
        except (ValueError, TypeError):
            return 0.0

class WallParameter(BaseModel):
    model_config = ConfigDict(extra="allow")
    room_name: Optional[str] = None
    length: Optional[Union[float, str]] = 0.0
    confidence: Optional[float] = Field(default=0.5, ge=0, le=1)
    width: Optional[float] = None
    height: Optional[Union[float, str]] = 0.0
    wall_type: Optional[str] = ""
    drywall_assembly: Optional[DrywallAssembly] = None
    code_references: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = None
    @field_validator("length", "height", mode="before")
    @classmethod
    def coerce_float(cls, v):
        if v is None:
            return 0.0
        try:
            return ensure_not_nan(float(v))
        except (ValueError, TypeError):
            return 0.0
    @field_validator("width", mode="before")
    @classmethod
    def coerce_optional_float(cls, v):
        if v is None:
            return v
        try:
            return ensure_not_nan(float(v))
        except (ValueError, TypeError):
            return None

class DrywallPredictorCaliforniaResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    ceiling: Optional[Ceiling] = None
    wall_parameters: List[WallParameter] = Field(default_factory=list)

SCALE_AND_CEILING_HEIGHT_DETECTOR = """
  You are an expert architectural drawing text parser

  PROVIDED:
    1. Cropped images from a floor plan that contains textual description notes.

  TASK:
    Identify the standard `ceiling_height` and `scale` mentioned in the transcription entries for the subsequent floorplan.
    INSTRUCTIONS:
      - Look for a keyword that matches with `ceiling height` field and identify the numerical entity closest to it. Note the feet equivalent of it.
      - Look for a keyword that has to do with the `scale` of the drawing, representing the ratio between the length on paper and the real world length in floating point values. Normalize and capture the ratio as "<paper_length_in_inches>``: <real_world_length_in_feet>`<real_world_length_in_inches>``".
          Example: 0.25``:1`0``
      - If multiple ceiling heights are listed, extract the standard or typical one.
      - If scale is written in multiple formats, preserve the exact textual format.
      - If not present, return null.

  OUTPUT:
    Your output should be in the JSON format containing the standard `ceiling_height` and `scale` of the floorplan.
    **STRICTLY** Do not generate additional content apart from the designated JSON.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
        "ceiling_height": <Standard ceiling height mentioned in the transcriptions converted to feet in float>,
        "scale": "<Scale of the drawing mentioned in the transcriptions i.e. number_in_inches``: number_in_feet`number_in_inches``>"
    }}
"""

class ScaleAndCeilingHeightDetectorResponse(BaseModel):
    ceiling_height: Union[float, int]
    scale: str

CEILING_CHOICES = [
    "Flat",
    "Single-sloped",
    "Gable",
    "Tray",
    "Barrel vault",
    "Coffered",
    "Combination",
    "Soffit",
    "Cove",
    "Dome",
    "Cloister Vault",
    "Knee-Wall",
    "Cathedral with Flat Center",
    "Angled-Plane",
    "Boxed-Beam"
]
