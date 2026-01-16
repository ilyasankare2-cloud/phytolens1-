from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base

class Scan(Base):
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    image_path = Column(String, nullable=False)
    # Store the full analysis result as JSON
    result_json = Column(JSON, nullable=True)
    # Store primary classification for quick query
    primary_classification = Column(String, index=True, nullable=True) 
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="scans")

# Add relationship to User model (Need to modify user.py or do it here if possible, 
# but usually it's better to have back_populates in both if needed, 
# or just foreign key here is enough for MVP)
