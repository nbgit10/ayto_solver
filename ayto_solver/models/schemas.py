"""Pydantic schemas for AYTO API."""
from typing import List, Tuple, Dict, Optional
from pydantic import BaseModel, Field


class MatchingNight(BaseModel):
    """Data for a single matching night."""

    pairs: List[Tuple[str, str]] = Field(
        ...,
        description="List of (male, female) pairs matched that night",
        example=[["Alex", "Jules"], ["Tommy", "Melina"]]
    )
    matches: int = Field(
        ...,
        ge=0,
        description="Number of correct matches in this night",
        example=3
    )


class TruthBooth(BaseModel):
    """Data for a single truth booth result."""

    pair: Tuple[str, str] = Field(
        ...,
        description="The (male, female) pair tested",
        example=["Francesco", "Jules"]
    )
    match: bool = Field(
        ...,
        description="True if they are a perfect match, False otherwise",
        example=True
    )


class MatchInput(BaseModel):
    """Input data for solving AYTO matching problem."""

    males: List[str] = Field(
        ...,
        min_length=2,
        description="List of male contestant names",
        example=["Alex", "Tommy", "Francesco"]
    )
    females: List[str] = Field(
        ...,
        min_length=2,
        description="List of female contestant names",
        example=["Jules", "Melina", "Steffi"]
    )
    matching_nights: List[MatchingNight] = Field(
        default=[],
        description="List of matching night results"
    )
    truth_booths: List[TruthBooth] = Field(
        default=[],
        description="List of truth booth results"
    )


class MatchProbability(BaseModel):
    """Probability that a specific pair is a perfect match."""

    male: str
    female: str
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability this pair is a perfect match (0.0 to 1.0)"
    )


class DoublMatchCandidate(BaseModel):
    """Candidate for being involved in a double match (n×m scenarios)."""

    name: str
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability this person is involved in double match"
    )
    gender: str = Field(
        ...,
        description="'male' or 'female'"
    )


class SingleSolution(BaseModel):
    """A single matching solution."""

    matches: List[Tuple[str, str]] = Field(
        ...,
        description="List of (male, female) pairs in this solution"
    )


class MIPSolutionResponse(BaseModel):
    """Response from MIP solver endpoint."""

    solution: SingleSolution = Field(
        ...,
        description="A valid matching solution"
    )
    num_males: int
    num_females: int
    total_matches: int
    solver_type: str = "mip"

    # Enhanced fields (populated when enumerate_solutions=True)
    match_probabilities: Optional[List[MatchProbability]] = Field(
        default=None,
        description="Probabilities for each pair (only if enumeration enabled)"
    )
    double_match_candidates: Optional[List[DoublMatchCandidate]] = Field(
        default=None,
        description="For n×m cases, who might be in double match (only if enumeration enabled)"
    )
    total_solutions: Optional[int] = Field(
        default=None,
        description="Total number of solutions found (only if enumeration enabled)"
    )
    solutions_capped: Optional[bool] = Field(
        default=None,
        description="True if solution enumeration hit the cap (only if enumeration enabled)"
    )

    # Single solution double match info (always populated for n×m)
    double_match_in_solution: Optional[List[Tuple[str, str]]] = Field(
        default=None,
        description="In this specific solution, which person(s) have double matches: [(person, partner1), (person, partner2)]"
    )


class GraphSolutionResponse(BaseModel):
    """Response from graph-based solver endpoint."""

    match_probabilities: List[MatchProbability] = Field(
        ...,
        description="Probabilities for each possible pair"
    )
    double_match_candidates: Optional[List[DoublMatchCandidate]] = Field(
        default=None,
        description="For n×m cases, who might be in double match"
    )
    total_solutions: int = Field(
        ...,
        description="Total number of valid solutions found"
    )
    solutions_capped: bool = Field(
        ...,
        description="True if solution enumeration hit the cap (1000)"
    )
    num_males: int
    num_females: int
    solver_type: str = "graph"
    example_solutions: Optional[List[SingleSolution]] = Field(
        default=None,
        description="Up to 5 example solutions"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    details: Optional[str] = None
