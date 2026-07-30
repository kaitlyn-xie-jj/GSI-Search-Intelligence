import math
from dataclasses import dataclass, field
from typing import List

@dataclass
class TaskReqGeq:
    cap_id: int
    cap_req: float

    def to_string(self) -> str:
        operator = ">"
        result = f"cap{self.cap_id}{operator}{self.cap_req:.2f}"
        return f"{result}"

@dataclass
class TaskParam:
    req_fcn: List[List[TaskReqGeq]] = field(default_factory=list)
    time_cost: float = 0.0

    def to_string(self) -> str:
        clauses = []
        for and_clause in self.req_fcn:
            or_clauses = " || ".join(str(req.to_string()) for req in and_clause)
            clauses.append(f"({or_clauses})")
        
        and_clauses = " & ".join(clauses)
        final_clauses = and_clauses + f", time cost: {self.time_cost}"
        return final_clauses
    
if __name__=="__main__":
    req1 = TaskReqGeq(cap_id=1, cap_req=5.0)
    req2 = TaskReqGeq(cap_id=2, cap_req=3.5)

    task_param = TaskParam()
    task_param.req_fcn = [
        [req1, TaskReqGeq(3, 2.0)], 
        [req2]                         
    ]

    print(task_param.to_string())