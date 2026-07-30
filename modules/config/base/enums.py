"""Global Enumeration Definitions Module

Defines all enumeration types used in the system to ensure global consistency.
"""
from enum import Enum
from typing import List

# Top-level categories
class Category(Enum):
    BUILDING = "building"
    AREA = "area"
    TRANS_Facility = "trans_facility"
    POI = "poi"
    PROP = "prop"
    ROBOT = "robot"
    DISTRICT = "district"

# Areas
class AreaType(Enum):
    GARDEN = "garden"
    SQUARE = "square"
    WATER_BODY = "water_body"
    GREENBELT = "greenbelt"
    INDUSTRIAL_PARK = "industrial_park"
    CAMPUS = "campus"
    NEIGHBORHOOD = "neighborhood"

# Transportation facilities
class TransFacilityType(Enum):
    INTERSECTION = "intersection"
    STREET_SEGMENT = "street_segment"
    SIDEWALK = "sidewalk"
    BRIDGE = "bridge"

class POIType(Enum):
    CHARGING_STATION = "charging_station"
    BUS_STOP = "bus_stop"
    ENTRANCE = "entrance"
    EXIT = "exit"

# Buildings
class BuildingName(Enum):
    HOSPITAL = "hospital"
    POWER_STATION = "power_station"
    MALL = "mall"
    PARKING = "parking"
    LIBRARY = "library"
    HOTEL = "hotel"
    ROBOT_BASE = "robot_base"

class BuildingStatus(Enum):
    DISCOVERED = "discovered"        
    UNDISCOVERED = "undiscovered"  

# Robots
class RobotName(Enum):
    UAV = "UAV"
    FW_UAV = "FW_UAV"
    UGV = "UGV"
    Quadruped = "Quadruped" 
    Humanoid = "Humanoid"  

class RobotAttributeName(Enum):
    BATTERY_LEVEL = "battery_level"
    ALTITUDE = "altitude"
    CURRENT_LOAD = "current_load"
    MAX_PAYLOAD_KG = "max_payload_kg"
    MAX_SPEED_MS = "max_speed_ms"

class RobotStatus(Enum):
    IDLE = "idle"                    
    MOVING = "moving"               
    WORKING = "working"              
    CHARGING = "charging"            
    ERROR = "error"                          

# Props
class PropName(Enum):
    PERSON = "person" 
    VEHICLE = "vehicle"
    BOAT = "boat"
    FIRE = "fire"
    CARGO = "cargo"
    HAZMAT = "hazmat"
    EQUIPMENT_FAILURE = "equipment_failure"
    ASSEMBLY_COMPONENT = "assembly_component"

class PropAttributeName(Enum):
    LICENSE_PLATE = "license_plate"
    WEIGHT_KG = "weight_kg"
    COLOR = "color"                
    SUBTYPE = "subtype"             
    CLOTHING_COLOR = "clothing_color" 
    ITEM = "item" 

class PropStatus(Enum):
    AVAILABLE = "available"         
    UNAVAILABLE = "unavailable"      
    DISCOVERED = "discovered"        
    UNDISCOVERED = "undiscovered"    
    LOADED = "loaded"                
    UNLOADED = "unloaded"            
    IN_USE = "in_use"                
    RESOLVED = "resolved"              
    UNRESOLVED = "unresolved" 

class PersonItem(Enum):
    BACKPACK = "backpack"
    SUITCASE = "suitcase"
    HAT = "hat"
    UMBRELLA = "umbrella"
    HANDBAG = "handbag"
    CAMERA = "camera"
    
class CarSubtype(Enum):
    SEDAN = "sedan"
    SUV = "suv"
    TRUCK = "truck"
    VAN = "van"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"

class BoatSubtype(Enum):
    SPEEDBOAT = "speedboat"
    CARGO_SHIP = "cargo_ship"
    FISHING_BOAT = "fishing_boat"
    YACHT = "yacht"
    SAILBOAT = "sailboat" 

class CargoSubtype(Enum):
    BOX = "box"
    CRATE = "crate"
    MEDICAL_SUPPLY = "medical_supply"   
    FOOD_SUPPLY = "food_supply"         
    WATER_CONTAINER = "water_container" 
    TOOLKIT = "toolkit"
    BATTERY_PACK = "battery_pack"

class AssemblyComponentType(Enum):
    FOUNDATION_BASE = "foundation_base"
    WALL_PANEL = "wall_panel"
    ROOF_PANEL = "roof_panel"
    SOLAR_PANEL = "solar_panel"
    LIGHTING_UNIT = "lighting_unit"
    ANTENNA_MODULE = "antenna_module"
    PUBLIC_DISPLAY_SCREEN = "display_screen"
    PUBLIC_ADDRESS_SPEAKER = "address_speaker"
    WEATHER_STATION_MODULE = "weather_module"
    SURVEILLANCE_CAMERA_MAST = "surveillance_mast"
    ROBOT_CHARGING_DOCK = "charging_dock"
    DRONE_LANDING_PAD = "landing_pad"
    SMART_TRASH_RECEPTACLE = "smart_trash_receptacle"
    PUMP_MODULE = "pump_module"
    EMERGENCY_CALL_BOX = "emergency_box"

# Skills
class SkillName(Enum):
    TAKE_OFF = "take_off"            
    LAND = "land"                    
    RETURN_HOME = "return_home"   
    NAVIGATE = "navigate"            
    TAKE_PHOTO = "take_photo"  
    BROADCAST = "broadcast"      
    HANDLE_HAZARD = "handle_hazard"
    SEARCH = "search"    
    FOLLOW = "follow"
    GUIDE = "guide"
    PLACE = "place"

# New situations
class NewSituationType(Enum):
    # Robot
    ROBOT_ADD = "robot_add"
    ROBOT_REMOVE = "robot_remove"
    ROBOT_UPDATE = "robot_update"
    # Building
    BUILDING_ADD = "building_add"
    BUILDING_REMOVE = "building_remove"
    BUILDING_UPDATE = "building_update"
    # Prop
    PROP_ADD = "prop_add"
    PROP_REMOVE = "prop_remove"
    PROP_UPDATE = "prop_update"
    # Goal
    GOAL_ADD = "goal_add"
    GOAL_REMOVE = "goal_remove"
    GOAL_UPDATE = "goal_update"

# General-purpose enums
class ColorOption(Enum):
    RED = "red"
    WHITE = "white"
    BLACK = "black"
    BLUE = "blue"
    SILVER = "silver"
    GRAY = "gray"
    GREEN = "green"
    YELLOW = "yellow"
    PURPLE = "purple"
    ORANGE = "orange"

# Other
class RunMode(Enum):
    """Run mode enumeration."""
    DEFAULT = "default"
    LLM_FINETUNE = "llm_finetune"
    
class EventName(Enum):
    ILLEGAL_PARKING="illegal_parking"
    TRAFFIC_VIOLATION="traffic_violation"
    CROWD="crowd"
    CROWD_GATHERING="crowd_gathering"

class EdgeType(Enum):
    LOCATED_IN = "located_in"
    LOCATED_AT = "located_at"
    STATIONED_AT = "stationed_at"
    STORED_AT = "stored_at"
    CONNECTS_TO = "connects_to"
    ADJACENT_TO = "adjacent_to"
    FRONTS_ON = "fronts_on"
    HAS_ENTRANCE = "has_entrance"
    TRAVERSABLE = "traversable"

class CoordinationType(Enum):
    NONE = "none"
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"

class PositionAttributeName(Enum):
    """Position attribute name enumeration."""
    X = "x"
    Y = "y"
    Z = "z"
    ROTATION = "rotation"
    ORIENTATION = "orientation"

class ShapeType(Enum):
    """Shape type enumeration."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    POLYGON = "polygon"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    CROSS = "cross"
    OVAL = "oval"
    HEXAGON = "hexagon"

class RenderStyle(Enum):
    """Render style enumeration."""
    SOLID = "solid"
    OUTLINE = "outline"
    DASHED = "dashed"
    DOTTED = "dotted"
    GRADIENT = "gradient"
    PATTERN = "pattern"

class ColorScheme(Enum):
    """Color scheme enumeration."""
    BLUE = "#2196F3"
    GREEN = "#4CAF50"
    ORANGE = "#FF9800"
    RED = "#F44336"
    PURPLE = "#9C27B0"
    TEAL = "#009688"
    BROWN = "#795548"
    GRAY = "#607D8B"
    PINK = "#E91E63"
    INDIGO = "#3F51B5"
    CYAN = "#00BCD4"

class Quantifier(Enum):
    """Quantifier enumeration."""
    EXISTS = "EXISTS"       # Existential (at least one)
    FORALL = "FORALL"      # Universal (all)

class Operator(Enum):
    """Condition operator enumeration."""
    EQ = "EQ"              # Equal to
    NEQ = "NEQ"            # Not equal to
    GT = "GT"              # Greater than
    GTE = "GTE"            # Greater than or equal to
    IN = "IN"              # Contained in

class OutcomeType(Enum):
    """Skill execution outcome type enumeration."""
    KNOWLEDGE_ACQUIRED = "KNOWLEDGE_ACQUIRED"
    AREA_COVERAGE_UPDATE = "AREA_COVERAGE_UPDATE"
    PROCESS_STEP_COMPLETED = "PROCESS_STEP_COMPLETED"
    ENTITY_STATE_CHANGED = "ENTITY_STATE_CHANGED"
