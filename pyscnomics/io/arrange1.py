"""
Configurations to support data input preparation.
"""

from dataclasses import dataclass, field, InitVar
import numpy as np


@dataclass
class TaxSplit:
    """
    Prepare tax split data (PSC Cost Recovery).

    Parameters
    ----------
    goi_oil: float
        Government intake for oil.
    ctr_oil: float
        Contractor intake for oil.
    goi_gas: float
        Government intake for gas.
    ctr_gas: float
        Contractor intake for gas.
    """

    goi_oil: float
    ctr_oil: float
    goi_gas: float
    ctr_gas: float


@dataclass
class FTP:
    """
    Prepare FTP data (PSC Cost Recovery).

    Parameters
    ----------
    available: str
        Whether the FTP is available.
    contractor_shared: str
        Whether the FTP is shared with contractor.
    portion: float
        The percentage of FTP.
    """

    available: str
    contractor_shared: str
    portion: float


@dataclass
class CR:
    """
    Prepare cost recovery data (PSC Cost Recovery).

    Parameters
    ----------
    oil_capped: str
        Whether oil is capped.
    oil_costrec: float
        The percentage of oil cost recovery.
    gas_capped: str
        Whether gas is capped.
    gas_costrec: float
        The percentage of gas cost recovery.
    """

    oil_capped: str
    oil_costrec: float
    gas_capped: str
    gas_costrec: float


@dataclass
class Tax:
    """
    Prepare tax data (PSC Cost Recovery & Gross Split).

    Parameters
    ----------
    corporate_income: float
        Corporate income tax.
    branch_profit: float
        Branch profit tax.
    effective_rate: float
        Effective tax rate.
    """

    corporate_income: float
    branch_profit: float
    effective_rate: float


@dataclass
class DMO:
    """
    Prepare DMO data (PSC Cost Recovery & Gross Split).

    Parameters
    ----------
    holiday: str
        DMO holiday.
    period: int
        DMO period.
    start_production: int
        DMO start production.
    volume: float
        DMO volume.
    fee: float
        DMO fee.
    """

    holiday: str
    period: int
    start_production: int
    volume: float
    fee: float


@dataclass
class InvestmentCredit:
    """
    Prepare investment credit data (PSC Cost Recovery & Gross Split).

    Parameters
    ----------
    oil_is_available: str
        Availability of investment credit (for oil).
    oil_ic: float
        The value of investment credit (for oil).
    gas_is_available: str
        Availability of investment credit (for gas).
    gas_ic: float
        The value of investment credit (for gas).
    """

    oil_is_available: str
    oil_ic: float
    gas_is_available: str
    gas_ic: float


@dataclass
class Incentive:
    """
    Prepare incentive variable data (PSC Cost Recovery & Gross Split).

    Parameters
    ----------
    vat_categ: str
        Category of VAT/PPN.
    vat_percent: float
        The percentage of VAT/PPN.
    vat_discount: float
        The value of VAT/PPN discount.
    lbt_discount: float
        The value of LBT/PBB discount.
    pdrd: float
        The percentage value of PDRD.
    pdrd_discount: float
        The value of PDRD discount.
    pdri: float
        The percentage of PDRI.
    pdri_discount: float
        The value of PDRI discount.
    import_duty: float
        The percentage value of import duty (pajak bea masuk).
    vat_import: float
        The percentage value of import VAT/PPN.
    pph_import: float
        The percentage value of import PPH.
    depre_accel: str
        Whether depreciation is accelerated.
    depre_method: str
        Depreciation method.
    """

    vat_categ: str
    vat_percent: float
    vat_discount: float
    lbt_discount: float
    pdrd: float
    pdrd_discount: float
    pdri: float
    pdri_discount: float
    import_duty: float
    vat_import: float
    pph_import: float
    depre_accel: str
    depre_method: str


@dataclass
class VarSplitSet:
    """
    Capture the name of the property and its corresponding value.

    Parameters
    ----------
    param: str
        The name of a particular property.
    val: float
        The value of a particular property.
    """

    param: str
    val: float


@dataclass
class VariableSplit:
    """
    Prepare data related to variable split.

    Parameters
    ----------
    field_status: VarSplitSet
        Status of the field.
    field_loc: VarSplitSet
        Field's location.
    res_depth: VarSplitSet
        Reservoir depth.
    infra_avail: VarSplitSet
        Infrastructure availability.
    res_type: VarSplitSet
        Reservoir type.
    api_oil: VarSplitSet
        API oil.
    domestic_use: VarSplitSet
        Domestic use (tingkat komponen dalam negeri).
    prod_stage: VarSplitSet
        Production stage.
    co2_content: VarSplitSet
        CO2 content.
    h2s_content: VarSplitSet
        H2S content.
    total_split: float
        Total split.
    """

    field_status: VarSplitSet
    field_loc: VarSplitSet
    res_depth: VarSplitSet
    infra_avail: VarSplitSet
    res_type: VarSplitSet
    api_oil: VarSplitSet
    domestic_use: VarSplitSet
    prod_stage: VarSplitSet
    co2_content: VarSplitSet
    h2s_content: VarSplitSet
    total_split: float


@dataclass
class SplitConfig:
    """
    Prepare data associated with split configuration.

    Parameters
    ----------
    ctr_oil: float
        Base split for contractor (oil).
    ctr_gas: float
        Base split for contractor (gas).
    ministry_discr: float
        Split due to ministry discretion.
    """

    ctr_oil: float
    ctr_gas: float
    ministry_discr: float


@dataclass
class CostElement:
    """
    Prepare the attributes of cost elements.

    Parameters
    ----------
    project_years: np.ndarray
        The array of years representing active project.
    cost_categ: str
        Cost component (Sunk Cost, Drilling Tangible, Drilling Intangible, Facility Tangible,
        Operating Cost, ASR, Land and Building Tax, PDRD, Instangible Asset)
    target_cost: np.ndarray
        Target data input to be prepared.
    """

    project_years: np.ndarray
    cost_categ: str = field(repr=False)
    target_cost: np.ndarray = field(repr=False)

    # Attributes to be defined later in __post_init__
    cost_pre_vat: np.ndarray = field(default=None, init=False)
    cost_post_vat: np.ndarray = field(default=None, init=False)
    year_expensed: np.ndarray = field(default=None, init=False)
    pis: np.ndarray = field(default=None, init=False)
    vat_portion: np.ndarray = field(default=None, init=False)
    pdri_portion: np.ndarray = field(default=None, init=False)

    def __post_init__(self):

        # Initiate attributes as zeros
        self.id_same_year = np.zeros(self.target_cost.shape[0], dtype=int)
        self.cost_pre_vat = np.zeros(len(self.project_years), dtype=np.float_)
        self.cost_post_vat = np.zeros(len(self.project_years), dtype=np.float_)
        self.vat_portion = np.zeros(len(self.project_years), dtype=np.float_)
        self.pdri_portion = np.zeros(len(self.project_years), dtype=np.float_)

        for i in range(self.target_cost.shape[0]):

            # For cost component drilling and facility tangible
            if self.cost_categ == "Drilling Tangible" or self.cost_categ == "Facility Tangible":

                # PIS data is available
                if int(self.target_cost[i, 3]) != 0:

                    # Index row where the 'PIS' is equal to the correponding project year
                    self.id_same_year[i] = np.argwhere(self.target_cost[i, 3] == self.project_years).ravel()

                    # Modify attributes at the specified rows(s)
                    self.cost_pre_vat[self.id_same_year[i]] = self.target_cost[i, 0]
                    self.cost_post_vat[self.id_same_year[i]] = self.target_cost[i, 1]
                    self.vat_portion[self.id_same_year[i]] = self.target_cost[i, 4]
                    self.pdri_portion[self.id_same_year[i]] = self.target_cost[i, 5]

                # PIS data is not available
                elif int(self.target_cost[i, 3]) == 0:

                    # Year expensed data is not available
                    if self.target_cost[i, 2] == 0:
                        raise ValueError(f"PIS and year expensed data are missing")

                    # Year expensed data is available
                    else:

                        # Index row where the 'year expensed' is equal to the correponding project year
                        self.id_same_year[i] = np.argwhere(
                            self.target_cost[i, 2] == self.project_years
                        ).ravel()

                        # Modify attributes at the specified rows(s)
                        self.cost_pre_vat[self.id_same_year[i]] = self.target_cost[i, 0]
                        self.cost_post_vat[self.id_same_year[i]] = self.target_cost[i, 1]
                        self.vat_portion[self.id_same_year[i]] = self.target_cost[i, 4]
                        self.pdri_portion[self.id_same_year[i]] = self.target_cost[i, 5]

            # For other cost components
            else:

                # Year expensed data is not available
                if self.target_cost[i, 2] == 0:
                    raise ValueError(f"Year expensed data is missing")

                # Year expensed data is available
                else:

                    # Index row where the 'year expensed' is equal to the correponding project year
                    self.id_same_year[i] = np.argwhere(self.target_cost[i, 2] == self.project_years).ravel()

                    # Modify attributes at the specified rows(s)
                    self.cost_pre_vat[self.id_same_year[i]] = self.target_cost[i, 0]
                    self.cost_post_vat[self.id_same_year[i]] = self.target_cost[i, 1]
                    self.vat_portion[self.id_same_year[i]] = self.target_cost[i, 4]
                    self.pdri_portion[self.id_same_year[i]] = self.target_cost[i, 5]


@dataclass
class CostComponent:
    """
    Prepare the attributes of cost component.

    Parameters
    ----------
    project_years: np.ndarray
        The array of years representing active project.
    target_cost: np.ndarray
        Target data input to be prepared.
    """

    project_years: InitVar[np.ndarray] = field(repr=False)
    target_cost: InitVar[np.ndarray] = field(repr=False)

    # Attributes to be defined later in __post_init__
    sunk_cost: CostElement = field(default=None, init=False)
    drilling_tangible: CostElement = field(default=None, init=False)
    drilling_intangible: CostElement = field(default=None, init=False)
    facility_tangible: CostElement = field(default=None, init=False)
    operating_cost: CostElement = field(default=None, init=False)
    asr: CostElement = field(default=None, init=False)
    lbt: CostElement = field(default=None, init=False)
    pdrd: CostElement = field(default=None, init=False)
    intangible_asset: CostElement = field(default=None, init=False)

    def __post_init__(self, project_years, target_cost):

        # Instantiate 'sunk_cost' as an instance of CostElement class
        self.sunk_cost = CostElement(
            project_years=project_years,
            cost_categ="Sunk Cost",
            target_cost=target_cost["Sunk Cost"]
        )

        # Instantiate 'drilling_tangible' as an intance of CostElement class
        self.drilling_tangible = CostElement(
            project_years=project_years,
            cost_categ="Drilling Tangible",
            target_cost=target_cost["Drilling Tangible"]
        )

        # Instantiate 'drilling_intangible' as an instance of CostElement class
        self.drilling_intangible = CostElement(
            project_years=project_years,
            cost_categ="Drilling Intangible",
            target_cost=target_cost["Drilling Intangible"]
        )

        # Instantiate 'facility_tangible' as an instance of CostElement class
        self.facility_tangible = CostElement(
            project_years=project_years,
            cost_categ="Facility Tangible",
            target_cost=target_cost["Facility Tangible"]
        )

        # Instantiate 'operating_cost' as an instance of CostElement class
        self.operating_cost = CostElement(
            project_years=project_years,
            cost_categ="Operating Cost",
            target_cost=target_cost["Operating Cost"]
        )

        self.asr = CostElement(
            project_years=project_years,
            cost_categ="ASR",
            target_cost=target_cost["ASR"]
        )

        self.lbt = CostElement(
            project_years=project_years,
            cost_categ="Land and Building Tax",
            target_cost=target_cost["Land and Building Tax"]
        )

        self.pdrd = CostElement(
            project_years=project_years,
            cost_categ="PDRD",
            target_cost=target_cost["PDRD"]
        )

        self.intangible_asset = CostElement(
            project_years=project_years,
            cost_categ="Intangible Asset",
            target_cost=target_cost["Intangible Asset"]
        )


@dataclass
class LiftingElement:
    """
    Prepare the attributes of lifting elements.

    Parameters
    ----------
    start_year: int
        The start year of the project.
    end_year: int
        The end year of the project.
    prod_rate: np.ndarray
        The production rate of a particular fluid type.
    price: np.ndarray
        The associated price of a particular fluid type.
    ghv: np.ndarray
        The value of ghv of a particular fluid type (default value = 1).
    days: np.ndarray
        The number of days used as reference to specify prod_rate.
    """

    start_year: int
    end_year: int
    prod_rate: np.ndarray = field(default=None)
    price: np.ndarray = field(default=None)
    ghv: np.ndarray = field(default=None)
    days: np.ndarray = field(default=None)

    # Attributes to be defined later on
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    def __post_init__(self):

        # Instantiate project duration and project years
        if self.start_year > self.end_year:
            raise ValueError(f"Start year {self.start_year} is after end year {self.end_year}")

        else:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        # Condition when user does not insert GHV data; the default value of GHV is set to unity
        if self.ghv is None:
            self.ghv = np.ones(self.project_duration)

        # Condition when user does not insert days of production; the default value is set to 365
        if self.days is None:
            self.days = np.ones(self.project_duration) * 365
