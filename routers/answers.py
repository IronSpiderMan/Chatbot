from fastapi import APIRouter

router = APIRouter(
    prefix="/answers",
    tags=["answers"],
    responses={404: {"description": "Not found"}}
)


@router.get('/')
async def answer(question: str):
    return {"question": question, "answer": "111"}
