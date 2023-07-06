from pydantic import BaseModel
from datetime import datetime


class History(BaseModel):
    id: int
    create_time: datetime
    question: str

    class Config:
        orm_mode = True


print(datetime)
