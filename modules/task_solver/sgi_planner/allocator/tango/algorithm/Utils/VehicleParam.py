from dataclasses import dataclass
from typing import List

@dataclass
class VehicleParam:
    eng_max: float = 0.0
    eng_cost: float = 0.0
    cap_vector: List[float] = None

    def to_string(self) -> str:
        components = [f"({self.eng_max:.2f}| "]
        components.append(")")
        
        return "".join(components)
    
if __name__ == "__main__":
    vehicle = VehicleParam()
    vehicle.eng_max = 2.5
    vehicle.cap_vector = [10.2, 20.7]
    print(vehicle) 