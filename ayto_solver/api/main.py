"""FastAPI application for AYTO solver."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import traceback

from ayto_solver.models.schemas import (
    MatchInput,
    MIPSolutionResponse,
    GraphSolutionResponse,
    SingleSolution,
    MatchProbability,
    DoublMatchCandidate,
    ErrorResponse,
)
from ayto_solver.solvers.mip_solver import MIPSolver
from ayto_solver.solvers.mip_multi_solver import MIPMultiSolver
from ayto_solver.solvers.graph_solver import GraphSolver


app = FastAPI(
    title="AYTO Solver API",
    description="Solve 'Are You The One' matching problems using MIP and graph-based algorithms",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        with open(index_path, "r") as f:
            return f.read()
    else:
        return """
        <html>
            <head>
                <title>AYTO Solver API</title>
            </head>
            <body>
                <h1>AYTO Solver API</h1>
                <p>Welcome to the Are You The One matching solver!</p>
                <ul>
                    <li><a href="/docs">API Documentation (Swagger)</a></li>
                    <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                </ul>
                <h2>Available Endpoints:</h2>
                <ul>
                    <li><code>POST /solve/mip</code> - Solve using MIP (single solution)</li>
                    <li><code>POST /solve/graph</code> - Solve using graph algorithm (all solutions + probabilities)</li>
                </ul>
            </body>
        </html>
        """


def _detect_double_match_in_solution(solution_matrix, males, females):
    """
    Detect which person(s) have double matches in a solution.

    Args:
        solution_matrix: Binary matrix of matches
        males: List of male names
        females: List of female names

    Returns:
        List of tuples [(person, partner1), (person, partner2), ...] or None
    """
    import numpy as np

    double_matches = []

    # Check males (rows)
    for i, male in enumerate(males):
        partners = []
        for j, female in enumerate(females):
            if solution_matrix[i, j] > 0.5:
                partners.append(female)
        if len(partners) > 1:
            for partner in partners:
                double_matches.append((male, partner))

    # Check females (columns)
    for j, female in enumerate(females):
        partners = []
        for i, male in enumerate(males):
            if solution_matrix[i, j] > 0.5:
                partners.append(male)
        if len(partners) > 1:
            for partner in partners:
                double_matches.append((female, partner))

    return double_matches if double_matches else None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/solve/mip", response_model=MIPSolutionResponse)
async def solve_mip(input_data: MatchInput, enumerate_solutions: bool = False):
    """
    Solve AYTO matching problem using MIP solver.

    Args:
        input_data: Problem constraints (males, females, matching nights, truth booths)
        enumerate_solutions: If True, enumerate multiple solutions and calculate probabilities (slower)

    Returns:
        Single solution (fast) or multiple solutions with probabilities (slower)
    """
    try:
        if enumerate_solutions:
            # Use multi-solver for enumeration and probabilities
            solver = MIPMultiSolver(input_data.males, input_data.females)

            # Add matching night constraints
            for night in input_data.matching_nights:
                solver.add_matching_night(night.pairs, night.matches)

            # Add truth booth constraints
            for tb in input_data.truth_booths:
                solver.add_truth_booth(tb.pair[0], tb.pair[1], tb.match)

            # Enumerate solutions
            solutions, capped = solver.enumerate_solutions(max_solutions=1000)

            if not solutions:
                raise HTTPException(
                    status_code=404,
                    detail="No valid solutions found with given constraints"
                )

            # Calculate probabilities
            match_probs = solver.calculate_probabilities(solutions)
            double_match_probs = solver.calculate_double_match_probabilities(solutions)

            # Format match probabilities
            match_probabilities = [
                MatchProbability(male=male, female=female, probability=prob)
                for (male, female), prob in sorted(
                    match_probs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

            # Format double match candidates
            double_match_candidates = None
            if double_match_probs:
                # Determine gender based on who can have double matches
                if solver.n_females > solver.n_males:
                    gender = "male"
                else:
                    gender = "female"

                double_match_candidates = [
                    DoublMatchCandidate(name=name, probability=prob, gender=gender)
                    for name, prob in sorted(
                        double_match_probs.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                ]

            # Use first solution as the primary solution
            primary_solution = solutions[0]
            matches = solver.get_matches()  # From last solve

            # Detect double match in primary solution
            double_match_in_solution = _detect_double_match_in_solution(
                primary_solution, solver.males, solver.females
            )

            return MIPSolutionResponse(
                solution=SingleSolution(matches=matches),
                num_males=solver.n_males,
                num_females=solver.n_females,
                total_matches=len(matches),
                solver_type="mip",
                match_probabilities=match_probabilities,
                double_match_candidates=double_match_candidates,
                total_solutions=len(solutions),
                solutions_capped=capped,
                double_match_in_solution=double_match_in_solution
            )

        else:
            # Fast single-solution mode
            solver = MIPSolver(input_data.males, input_data.females)

            # Add matching night constraints
            for night in input_data.matching_nights:
                solver.add_matching_night(night.pairs, night.matches)

            # Add truth booth constraints
            for tb in input_data.truth_booths:
                solver.add_truth_booth(tb.pair[0], tb.pair[1], tb.match)

            # Solve
            solver.solve(timeout_seconds=10)

            # Validate solution
            is_valid, errors = solver.validate_solution()
            if not is_valid:
                raise HTTPException(
                    status_code=500,
                    detail=f"Solver produced invalid solution: {', '.join(errors)}"
                )

            # Get matches
            matches = solver.get_matches()

            # Detect double match in this solution
            double_match_in_solution = _detect_double_match_in_solution(
                solver.X_binary, solver.males, solver.females
            )

            return MIPSolutionResponse(
                solution=SingleSolution(matches=matches),
                num_males=solver.n_males,
                num_females=solver.n_females,
                total_matches=len(matches),
                solver_type="mip",
                double_match_in_solution=double_match_in_solution
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}\n{traceback.format_exc()}"
        )


@app.post("/solve/graph", response_model=GraphSolutionResponse)
async def solve_graph(input_data: MatchInput):
    """
    Solve AYTO matching problem using graph-based algorithm.

    Returns all possible solutions (up to 1000) and match probabilities.
    """
    try:
        # Create graph solver
        solver = GraphSolver(input_data.males, input_data.females)

        # Add truth booth constraints
        for tb in input_data.truth_booths:
            solver.add_truth_booth(tb.pair[0], tb.pair[1], tb.match)

        # Add matching night constraints
        for night in input_data.matching_nights:
            solver.add_matching_night(night.pairs, night.matches)

        # Enumerate all matchings
        matchings, capped = solver.enumerate_all_matchings(max_matchings=1000)

        if not matchings:
            raise HTTPException(
                status_code=404,
                detail="No valid solutions found with given constraints"
            )

        # Calculate probabilities
        match_probs = solver.calculate_probabilities(matchings)
        double_match_probs = solver.calculate_double_match_probabilities(matchings)

        # Format match probabilities
        match_probabilities = [
            MatchProbability(male=male, female=female, probability=prob)
            for (male, female), prob in sorted(
                match_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        # Format double match candidates
        double_match_candidates = None
        if double_match_probs:
            # Determine gender based on who can have double matches
            if solver.n_females > solver.n_males:
                gender = "male"
            else:
                gender = "female"

            double_match_candidates = [
                DoublMatchCandidate(name=name, probability=prob, gender=gender)
                for name, prob in sorted(
                    double_match_probs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

        # Get example solutions (up to 5)
        example_solutions = None
        if matchings:
            example_solutions = [
                SingleSolution(matches=list(matching))
                for matching in matchings[:5]
            ]

        return GraphSolutionResponse(
            match_probabilities=match_probabilities,
            double_match_candidates=double_match_candidates,
            total_solutions=len(matchings),
            solutions_capped=capped,
            num_males=solver.n_males,
            num_females=solver.n_females,
            solver_type="graph",
            example_solutions=example_solutions
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}\n{traceback.format_exc()}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
