"""
Collection of functions to carry out some operations related to module costs.py
"""

import numpy as np


class TaxInflationException(Exception):
    """ Exception to be raised for an incorrect tax and inflation configurations """

    pass


def check_input(target_func, param: np.ndarray | float | int) -> np.ndarray:
    """
    Check and prepare input parameters for subsequent analysis.

    Parameters
    ----------
    target_func: Function
        A target function.

    param : numpy.ndarray or float or int
        Parameter to be checked and prepared for analysis.

    Returns
    -------
    numpy.ndarray
        A properly formatted parameter array for analysis.

    Notes:
    ------
    This function checks the input parameter 'param' and ensures it is properly
    prepared for the subsequent analysis.
    If 'param' is a float or int, it creates an array of the same length as
    'target_func' with all elements set to 'param'.
    If 'param' is a numpy.ndarray, it checks that its length matches the length
    of 'target_func'. If not, it raises an exception.
    """

    if not isinstance(param, (float, int, np.ndarray)):
        raise TaxInflationException(
            f"Input parameter must be given as a float, int, or numpy.ndarray, "
            f"not as a/an {param.__class__.__qualname__}"
        )

    else:
        if isinstance(param, np.ndarray):
            if len(param) != len(target_func):
                raise TaxInflationException(
                    f"Unequal length of arrays: "
                    f"{param.__class__.__qualname__}({param}): ({len(param)}), "
                    f"{target_func.__class__.__qualname__}({target_func}): ({len(target_func)})."
                )
            param_arr = param

        elif isinstance(param, (float, int)):
            param_arr = np.repeat(param, len(target_func))

    return param_arr


def get_cost_adjustment_by_tax(
    start_year: int,
    cost: np.ndarray,
    expense_year: np.ndarray,
    project_years: np.ndarray,
    tax_portion: np.ndarray,
    tax_rate: np.ndarray | float,
    tax_discount: float,
) -> np.ndarray:
    """
    Adjust cost based on tax scheme over the prescribed time period.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    cost : np.ndarray
        An array of costs to be adjusted.
    expense_year : np.ndarray
        An array specifying the expense years for each cost element.
    project_years : np.ndarray
        An array of project years.
    tax_portion: np.ndarray
        The portion of 'cost' that is subject to tax.
        Must be an array of length equals to the length of 'cost' array.
    tax_rate: np.ndarray | float
        The tax rate to apply. Can be a single value or an array.
    tax_discount: float
        The tax discount to apply.

    Returns
    -------
    cost_adjusted: np.ndarray
        An array of adjusted costs after applying tax scheme.

    Notes
    -----
    The core operations are as follows:
    (1) Create 'tax_rate_arr' by calling 'check_input()' function. If 'tax_rate'
        is given as an array, the length of the array must be consistent with
        the duration of the project. If it is given as a single value, then create
        an array of length equal to the project duration with all elements set equal
        to the single value,
    (2) Identify the index location of 'tax_rate' based on the associated 'expense_year',
    (3) Calculate the multiplier:
        mult = 1.0 + tax_portion * tax_rate_arr[tax_id] * (1.0 - tax_discount),
    (4) Apply cost adjustment = cost * mult.
    """

    tax_rate_arr = check_input(target_func=project_years, param=tax_rate)
    tax_rate_id = (expense_year - start_year).astype("int")

    return cost * (
        1.0 + tax_portion * tax_rate_arr[tax_rate_id] * (1.0 - tax_discount)
    )


def get_cost_adjustment_by_inflation(
    start_year: int,
    end_year: int,
    cost: np.ndarray,
    expense_year: np.ndarray,
    project_years: np.ndarray,
    year_inflation: np.ndarray,
    inflation_rate: np.ndarray | float,
) -> np.ndarray:
    """
    Adjusts costs for inflation over the project years.

    This function applies inflation adjustments to costs based on the specified years
    and inflation rates. It calculates a multiplier for each cost value, adjusting it
    by inflation rates across the period between `start_year` and the `expense_year`.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year: int
        The end year of the project.
    cost : np.ndarray
        The array of costs to be adjusted for inflation.
    expense_year : np.ndarray
        The year(s) when the expenses occur, corresponding to the provided costs.
    project_years : np.ndarray
        An array of years representing the project timeline for inflation adjustments.
    year_inflation : np.ndarray
        An array of reference year for inflation calculation.
    inflation_rate : np.ndarray or float
        The inflation rate(s) for the project period. If a single float is provided,
        it applies uniformly across all years; otherwise, an array of rates for
        each year should be given.

    Returns
    -------
    np.ndarray
        An array of costs adjusted for inflation over the specified project years.

    Notes
    -----
    The core operations are as follows:
    (1) Check 'inflation_rate' whether it is provided as an array or as a single value.
        If it is given as an array, the length of the array must be consistent with the
        duration of the project. If it is given as a single value, then create an array
        of length equal to project duration with all elements set equal to the single
        value.
    (2) Parameter 'id_start' identify the index location of 'year_inflation' in array
        'project_years'. The result is then added by unity. This parameter sets up
        the first index to slice the 'inflation_rate_arr',
    (3) Parameter 'id_end' configure the index location of 'expense_year' in array
        'project_years' based on the associated 'year_inflation'. The result is then
        added by unity. This parameter sets up the second index to slice the 'inflation_rate_arr',
    (4) Slice 'inflation_rate_arr' according to 'id_start' and 'id_end'. Add the results
        by unity, then multiple the associated elements.
    (5) Cost adjustment is undertaken by multiplication: 'cost' * 'mult'.
    """

    # Prepare attribute year_inflation
    if year_inflation is None:
        year_inflation = np.repeat(start_year, len(cost))

    else:
        if not isinstance(year_inflation, np.ndarray):
            raise TaxInflationException(
                f"Attribute year_inflation must be given as a numpy.ndarray, "
                f"not as a/an {year_inflation.__class__.__qualname__}"
            )

        if len(year_inflation) != len(cost):
            raise TaxInflationException(
                f"Unequal length of arrays: "
                f"cost: {len(cost)}, "
                f"year_inflation: {len(year_inflation)}"
            )

    year_inflation = year_inflation.astype(np.int64)

    # Raise an error: year_inflation is before the start year of the project
    if np.min(year_inflation) < start_year:
        raise TaxInflationException(
            f"year_inflation ({np.min(year_inflation)}) is before the start year "
            f"of the project ({start_year})"
        )

    # Raise an error: year_inflation is after the end year of the project
    if np.max(year_inflation) > end_year:
        raise TaxInflationException(
            f"year_inflation ({np.max(year_inflation)}) is after the end year "
            f"of the project ({end_year})"
        )

    # Create an array of inflation_rate
    inflation_rate_arr = check_input(target_func=project_years, param=inflation_rate)

    # Specify the start and end indices to slice the inflation rate array
    id_start = (
        np.array(
            [
                np.argwhere(year_inflation[i] == project_years).ravel()
                for i, _ in enumerate(year_inflation)
            ]
        ).ravel() + 1
    ).astype(np.int64)

    id_end = (
        (expense_year - year_inflation) + (year_inflation - start_year) + 1
    ).astype(np.int64)

    # Multipliers to adjust cost by inflation
    mult = np.array(
        [
            np.prod(1.0 + inflation_rate_arr[id_start[i]:id_end[i]])
            for i, _ in enumerate(id_start)
        ]
    )

    return cost * mult


def apply_cost_adjustment(
    start_year: int,
    cost: np.ndarray,
    expense_year: np.ndarray,
    project_years: np.ndarray,
    tax_portion: np.ndarray,
    tax_rate: np.ndarray | float,
    tax_discount: float,
    inflation_rate: np.ndarray | float,
    year_ref: np.ndarray,
) -> np.ndarray:
    """
    Adjusts cost based on inflation and tax scheme over the specified time period.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    cost : np.ndarray
        An array of costs to be adjusted.
    expense_year : np.ndarray
        An array specifying the expense years for each cost element.
    project_years : np.ndarray
        An array of project years.
    tax_portion: np.ndarray
        The portion of 'cost' that is subject to tax (VAT or LBT).
        Must be an array of length equals to the length of 'cost' array.
    tax_rate: np.ndarray | float
        The tax rate to apply (VAT or LBT). Can be a single value or an array.
    tax_discount: float
        The tax discount to apply (VAT or LBT).
    inflation_rate: np.ndarray | float
        The inflation rate to apply. Can be a single value or an array.
    year_ref : np.ndarray
        An array of reference year for inflation calculation.

    Returns
    -------
    cost_adjusted: np.ndarray
        An array of adjusted costs after applying inflation and tax scheme.

    Notes
    -----
    The core operations are as follows:
    (1) Apply cost adjustment due tax scheme by calling function get_cost_adjustment_by_tax().
    (2) Check 'inflation_rate' whether it is provided as an array or as a single value.
        If it is given as an array, the length of the array must be consistent with the
        duration of the project. If it is given as a single value, then create an array
        of length equal to project duration with all elements set equal to the single
        value.
    (3) Parameter 'id_start' identify the index location of 'year_ref' in array
        'project_years'. The result is then added by unity. This parameter sets up
        the first index to slice the 'inflation_rate_arr',
    (4) Parameter 'id_end' configure the index location of 'expense_year' in array
        'project_years' based on the associated 'year_ref'. The result is then added by unity.
        This parameter sets up the second index to slice the 'inflation_rate_arr',
    (5) Slice 'inflation_rate_arr' according to 'id_start' and 'id_end'. Add the results
        by unity, then multiple the associated elements.
    (6) Cost adjustment is undertaken by multiplication: 'cost_adjusted_by_tax' * 'mult'.
    """

    # Cost adjustment due to tax
    cost_adjusted = get_cost_adjustment_by_tax(
        start_year=start_year,
        cost=cost,
        expense_year=expense_year,
        project_years=project_years,
        tax_portion=tax_portion,
        tax_rate=tax_rate,
        tax_discount=tax_discount,
    )

    # Create array of inflation_rate
    inflation_rate_arr = check_input(target_func=project_years, param=inflation_rate)

    # Specify the start and end indices to slice the inflation_rate array
    id_start = np.array(
        [
            np.argwhere(year_ref[i] == project_years).ravel()[0] + 1
            for i, val in enumerate(year_ref)
        ]
    )

    id_end = (
        (expense_year - year_ref) + (year_ref - start_year) + 1
    ).astype("int")

    # Multipliers to adjust cost by inflation
    mult = np.array(
        [
            np.prod(1.0 + inflation_rate_arr[id_start[i]:id_end[i]])
            for i, val in enumerate(id_start)
        ]
    )

    return cost_adjusted * mult


def calc_indirect_tax(
    start_year: int,
    cost: np.ndarray,
    expense_year: np.ndarray,
    project_years: np.ndarray,
    tax_portion: np.ndarray,
    tax_rate: np.ndarray | float,
    tax_discount: np.ndarray,
) -> np.ndarray:
    """
    Calculate the indirect tax on project costs.

    This function calculates the indirect tax by applying the specified tax portion,
    tax rates, and tax discount to the project costs for the corresponding expense years.

    Parameters
    ----------
    start_year : int
        The start year of the project. Used to calculate the tax rate indices.
    cost : np.ndarray
        A NumPy array representing the project costs to which the indirect tax will be applied.
    expense_year : np.ndarray
        A NumPy array representing the years during which expenses occur. Used to
        match tax rates to the corresponding years.
    project_years: np.ndarray
        A NumPy array representing project years.
    tax_portion : np.ndarray
        A NumPy array representing the portion of the cost that is subject to tax.
    tax_rate : np.ndarray | float
        A NumPy array of tax rates for each year of the project.
    tax_discount : np.ndarray
        A discount factor applied to the tax, reducing the overall tax impact.

    Returns
    -------
    np.ndarray
        A NumPy array containing the indirect tax amounts for each cost, adjusted for
        the tax portion, tax rates, and tax discount.

    Notes
    -----
    -   The function computes the tax rate indices based on the difference between
        `expense_year` and `start_year`.
    -   The indirect tax is calculated by applying the tax rate to the `tax_portion`
        of the `cost`, adjusted by the `tax_discount`.
    -   The function calculates the indirect tax using the formula:
            indirect_tax = cost * (tax_portion * tax_rate * (1.0 - tax_discount))
    """

    # Prepare attribute tax rate
    if not isinstance(tax_rate, (np.ndarray, float)):
        raise TaxInflationException(
            f"Attribute tax rate must be given as a float or as a numpy.ndarray, "
            f"not as a/an {tax_rate.__class__.__qualname__}"
        )

    else:
        tax_rate_arr = check_input(target_func=project_years, param=tax_rate)

    # Specify tax rate id
    tax_rate_ids = (expense_year - start_year).astype(np.int64)

    # Calculate indirect tax
    return cost * (tax_portion * tax_rate_arr[tax_rate_ids] * (1.0 - tax_discount))


def calc_distributed_cost(
    cost: np.ndarray,
    expense_year: np.ndarray,
    final_year: np.ndarray,
    project_years: np.ndarray,
    project_duration: int,
) -> np.ndarray:
    """
    Distribute the cost evenly over the project duration based on the expense and final years.

    This function decomposes the total cost of each element across multiple years,
    from the expense year to the final year. It then distributes the decomposed values
    over the specified project years.

    Parameters
    ----------
    cost : np.ndarray
        The total cost for each element. Must be a 1D array of floats.
    expense_year : np.ndarray
        The year in which each cost element is incurred. Must be a 1D array of integers.
    final_year : np.ndarray
        The final year for each cost element to be distributed. Must be a 1D array of integers.
    project_years : np.ndarray
        The array of project years used to index the distribution. Must be a 1D array of integers.
    project_duration : int
        The total duration of the project in years.

    Returns
    -------
    np.ndarray
        A 2D array where each column represents the distributed cost of an element over
        the project duration, with rows representing years.

    Notes
    -----
    -   The total cost for each element is split evenly across the number of years from
        `expense_year` to `final_year`.
    -   The cost is distributed across the range of `project_years` starting at `expense_year`
        and ending at `final_year`.
    -   The function returns a zero-filled matrix for years before the expense year or after
        the final year for each element.
    """

    # Prepare attribute cost
    if not isinstance(cost, np.ndarray):
        raise TaxInflationException(
            f"Argument cost must be given as a numpy.ndarray, "
            f"not as a/an {cost.__class__.__qualname__}"
        )

    # Prepare attribute expense_year
    if not isinstance(expense_year, np.ndarray):
        raise TaxInflationException(
            f"Argument expense_year must be given as a numpy.ndarray, "
            f"not as a/an {expense_year.__class__.__qualname__}"
        )

    # Unequal length of arrays
    if len(cost) != len(expense_year):
        raise TaxInflationException(
            f"Unequal length of arrays: "
            f"Expected: {len(expense_year)}, "
            f"Given: {len(cost)}"
        )

    cost = cost.astype(np.float64)

    # Total number of years to split/decompose each cost elements
    years_to_split = (final_year - expense_year + 1).astype(int)

    # Decomposed values for each cost elements
    cost_split = cost / years_to_split

    # The start and end indices to distribute each cost elements
    id_start = np.array(
        [
            np.argwhere(expense_year[i] == project_years).ravel()
            for i, _ in enumerate(expense_year)
        ]
    ).ravel()

    id_end = (id_start + years_to_split).astype(int)

    # Distributed values for each cost elements
    distributed_cost = np.zeros([project_duration, len(expense_year)], dtype=np.float64)

    for i, _ in enumerate(expense_year):
        distributed_cost[id_start[i]:id_end[i], i] = cost_split[i]

    return distributed_cost
