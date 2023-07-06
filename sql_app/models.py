from sqlalchemy import Column, Integer, String, DateTime

from .database import Base


class History(Base):
    __tablename__ = "histories"

    id = Column(Integer, primary_key=True, index=True)
    create_time = Column(DateTime)
    question = Column(String)
