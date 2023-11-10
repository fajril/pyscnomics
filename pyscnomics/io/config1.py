"""
Prepare data input from a target Excel workbook.
"""

from dataclasses import dataclass, field, InitVar
import datetime as dt
import numpy as np

from ionomics.helper import (
    TaxSplit,
    Tax,
    DMO,
    InvestmentCredit,
    FTP,
    CR,
    Incentive,
    VarSplitSet,
    VariableSplit,
    SplitConfig,
    CostComponent,
    LiftingElement
)


@dataclass
class GeneralConfig:
    """
    Prepare general config data input.

    Parameters
    ----------
    start_year: int
        The start year of the project.
    end_year: int
        The end year of the project.
    contract_type: str
        Type of PSC contract.
    start_new_contract: dt
        The start date of the contract.
    year_new_contract: int
        The start year of the contract.
    discount_rate_start: int
        The start year of discount rate.
    project_duration: int
        The active project duration.
    contract_basis: str
        Basis of the PSC contract.
    inflation_rate_applied_to: str
        Subject of inflation rate.
    economic_limit: int
        The end year of economic feasibility.

    depreciation_config: np.ndarray
        Depreciation data.
    vat_config: np.ndarray
        VAT/PPN data.
    inflation_config: np.ndarray
        Inflation data.
    gas_conversion_factor: np.ndarray
        Gas conversion factor data.
    lbt_config: np.ndarray
        LBT/PBB data.
    pdrd_config: np.ndarray
        PDRD data.
    etr_config: np.ndarray
        ETR (Effective Tax Rate) data.
    rc_split: np.ndarray
        RC Split data.
    icp_sliding: np.ndarray
        ICP Sliding Scale data.
    """

    start_year: int
    end_year: int
    contract_type: str
    start_new_contract: dt
    year_new_contract: int
    discount_rate_start: int
    project_duration: int = field(init=False)
    contract_basis: str = field(default='Stand Alone')
    inflation_rate_applied_to: str = field(default=None)
    economic_limit: str = field(default=None)

    depreciation_config: np.ndarray = field(default=None, repr=False)
    vat_config: np.ndarray = field(default=None, repr=False)
    inflation_config: np.ndarray = field(default=None, repr=False)
    gas_conversion_factor: np.ndarray = field(default=None, repr=False)
    lbt_config: np.ndarray = field(default=None, repr=False)
    pdrd_config: np.ndarray = field(default=None, repr=False)
    etr_config: np.ndarray = field(default=None, repr=False)
    rc_split: np.ndarray = field(default=None, repr=False)
    icp_sliding: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):

        # Define an attribute depicting project duration;
        # Raise a "ValueError" if start_year is after then end_year
        if self.end_year > self.start_year:
            self.project_duration = self.end_year - self.start_year + 1

        else:
            raise ValueError(
                f"start year {self.start_year} is after the end year: {self.end_year}"
            )

    @staticmethod
    def _create_array(data_config: np.ndarray) -> np.ndarray:
        """
        Create an array of two columns: the first column consists of years, the second column
        comprises a particular rate at the corresponding year.

        Parameters
        ----------
        data_config: np.ndarray
            A particular data upon which we create the array.

        Returns
        -------
        data_array: np.ndarray
            A two-column array of (year, rate).
        """

        # Configure the minimum and maximum years
        min_year = int(min(data_config[:, 1]))
        max_year = int(max(data_config[:, 2]))

        def _set_array(init_year: int, end_year: int, rate: float, array: np.ndarray) -> np.ndarray:
            """
            Fill the element of array with its corresponding rate.

            Parameters
            ----------
            init_year: int
                The initial year of rate implementation.
            end_year: int
                The final year of rate implementation.
            rate: float
                A particular rate.
            array: np.ndarray
                A two-column array of (year, rate).

            Returns
            -------
            array: np.ndarray
                A two-column array of (year, rate).
            """

            # Determine the initial and final indices of array that satisfies a condition
            init_id = int(np.argwhere(init_year == array[:, 0]).ravel())
            end_id = int(init_id + (end_year - init_year + 1))

            # Fill the corresponding indices of the array
            array[init_id:end_id, 1] = np.ones(int(end_year - init_year + 1)) * rate

            return array

        # Specify iterable arguments
        init_years = data_config[:, 1]
        end_years = data_config[:, 2]
        rates = data_config[:, 0]

        # Initiate array
        array = np.zeros([max_year - min_year + 1, 2])
        array[:, 0] = np.arange(min_year, max_year + 1, 1)

        # Execute function _set_array to fill the elements of array
        for i, j, k in zip(range(len(init_years)), range(len(end_years)), range(len(rates))):
            array = _set_array(
                init_year=init_years[i],
                end_year=end_years[j],
                rate=rates[k],
                array=array
            )

        return array

    @property
    def vat(self) -> np.ndarray:
        """
        Create an array of VAT (Value-Added Tax) rate.

        Returns
        -------
        vat_array: np.ndarray
            An array of VAT rate.
        """

        return self._create_array(data_config=self.vat_config)

    @property
    def inflation(self) -> np.ndarray:
        """
        Create an array of inflation rate.

        Returns
        -------
        inflation_array: np.ndarray
            An array of inflation rate.
        """

        return self._create_array(data_config=self.inflation_config)

    @property
    def lbt(self) -> np.ndarray:
        """
        Create an array of LBT (Land and Building Tax - Pajak Bumi dan Bangunan) rate.

        Returns
        -------
        lbt_array: np.ndarray
            An array of LBT/PBB rate.
        """

        return self._create_array(data_config=self.lbt_config)

    @property
    def pdrd(self) -> np.ndarray:
        """
        Create an array of PDRD rate.

        Returns
        -------
        pdrd_rate: np.ndarray
            An array of PDRD rate.
        """

        return self._create_array(data_config=self.pdrd_config)

    @property
    def etr(self) -> np.ndarray:
        """
        Create an array of ETR (Effective Tax Rate).

        Returns
        -------
        etr_rate: np.ndarray
            An array of ETR.
        """

        return self._create_array(data_config=self.etr_config)


@dataclass
class CostRecData:
    """
    Prepare PSC Cost Recovery data input.

    Parameters
    ----------
    split_config: str
        Split configuration.
    goi_tax: dict
        Government intake (pre- and post-tax; oil and gas).
    ctr_tax: dict
        Contractor intake (pre- and post-tax; oil and gas).
    corporate_income: float
        Corporate income tax.
    branch_profit: float
        Branch profit tax.
    effective_rate: float
        Effective tax rate.
    dmo_holiday: dict
        DMO holiday (for oil and gas).
    dmo_period: dict
        DMO period (for oil and gas).
    dmo_start_production: dict
        DMO start production (for oil and gas).
    dmo_volume: dict
        DMO volume (for oil and gas).
    dmo_fee: dict
        DMO fee (for oil and gas).
    ic_available: dict
        Availability of investment credit (for oil and gas).
    ic_value: dict
        The value of investment credit (for oil and gas).
    ftp_available: str
        Availability of FTP.
    ftp_ctr_shared: str
        Whether the FTP is shared with contractor.
    ftp_portion: float
        The percentage of FTP.
    is_capped: dict
        Whether the hydrocarbon is capped (for oil and gas).
    cr: dict
        The percentage of cost recovery (for oil and gas).
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

    # Properties used only for initialization (not assigned as class attributes)
    split_config: str
    goi_tax: InitVar[dict] = field(repr=False)
    ctr_tax: InitVar[dict] = field(repr=False)
    corporate_income: InitVar[float] = field(repr=False)
    branch_profit: InitVar[float] = field(repr=False)
    effective_rate: InitVar[float] = field(repr=False)
    dmo_holiday: InitVar[dict] = field(repr=False)
    dmo_period: InitVar[dict] = field(repr=False)
    dmo_start_production: InitVar[dict] = field(repr=False)
    dmo_volume: InitVar[dict] = field(repr=False)
    dmo_fee: InitVar[dict] = field(repr=False)
    ic_available: InitVar[dict] = field(repr=False)
    ic_value: InitVar[dict] = field(repr=False)
    ftp_available: InitVar[str] = field(repr=False)
    ftp_ctr_shared: InitVar[str] = field(repr=False)
    ftp_portion: InitVar[float] = field(repr=False)
    is_capped: InitVar[dict] = field(repr=False)
    cr: InitVar[dict] = field(repr=False)
    vat_categ: InitVar[str] = field(repr=False)
    vat_percent: InitVar[float] = field(repr=False)
    vat_discount: InitVar[float] = field(repr=False)
    lbt_discount: InitVar[float] = field(repr=False)
    pdrd: InitVar[float] = field(repr=False)
    pdrd_discount: InitVar[float] = field(repr=False)
    pdri: InitVar[float] = field(repr=False)
    pdri_discount: InitVar[float] = field(repr=False)
    import_duty: InitVar[float] = field(repr=False)
    vat_import: InitVar[float] = field(repr=False)
    pph_import: InitVar[float] = field(repr=False)
    depre_accel: InitVar[str] = field(repr=False)
    depre_method: InitVar[str] = field(repr=False)

    # The assigned attributes of the class instance
    post_tax: TaxSplit = field(default=None, init=False)
    pre_tax: TaxSplit = field(default=None, init=False)
    tax: Tax = field(default=None, init=False)
    dmo_oil: DMO = field(default=None, init=False)
    dmo_gas: DMO = field(default=None, init=False)
    ic: InvestmentCredit = field(default=None, init=False)
    ftp: FTP = field(default=None, init=False)
    cost_recovery: CR = field(default=None, init=False)
    incentive: Incentive = field(default=None, init=False)

    def __post_init__(self,
                      goi_tax,
                      ctr_tax,
                      corporate_income,
                      branch_profit,
                      effective_rate,
                      dmo_holiday,
                      dmo_period,
                      dmo_start_production,
                      dmo_volume,
                      dmo_fee,
                      ic_available,
                      ic_value,
                      ftp_available,
                      ftp_ctr_shared,
                      ftp_portion,
                      is_capped,
                      cr,
                      vat_categ,
                      vat_percent,
                      vat_discount,
                      lbt_discount,
                      pdrd,
                      pdrd_discount,
                      pdri,
                      pdri_discount,
                      import_duty,
                      vat_import,
                      pph_import,
                      depre_accel,
                      depre_method
                      ):
        # Create an instance of post tax
        self.post_tax = TaxSplit(
            goi_oil=goi_tax["post_tax_oil"],
            ctr_oil=ctr_tax["post_tax_oil"],
            goi_gas=goi_tax["post_tax_gas"],
            ctr_gas=ctr_tax["post_tax_gas"]
        )

        # Create an instance of pre tax
        self.pre_tax = TaxSplit(
            goi_oil=goi_tax["pre_tax_oil"],
            ctr_oil=ctr_tax["pre_tax_oil"],
            goi_gas=goi_tax["pre_tax_gas"],
            ctr_gas=ctr_tax["pre_tax_gas"]
        )

        # Create an instance of tax
        self.tax = Tax(
            corporate_income=corporate_income,
            branch_profit=branch_profit,
            effective_rate=effective_rate
        )

        # Create an instance of DMO oil
        self.dmo_oil = DMO(
            holiday=dmo_holiday["oil"],
            period=dmo_period["oil"],
            start_production=dmo_start_production["oil"],
            volume=dmo_volume["oil"],
            fee=dmo_fee["oil"]
        )

        # Create an instance of DMO gas
        self.dmo_gas = DMO(
            holiday=dmo_holiday["gas"],
            period=dmo_period["gas"],
            start_production=dmo_start_production["gas"],
            volume=dmo_volume["gas"],
            fee=dmo_fee["gas"]
        )

        # Create an instance of investment credit
        self.ic = InvestmentCredit(
            oil_is_available=ic_available["oil"],
            oil_ic=ic_value["oil"],
            gas_is_available=ic_available["gas"],
            gas_ic=ic_value["gas"]
        )

        # Create an instance of ftp
        self.ftp = FTP(
            available=ftp_available,
            contractor_shared=ftp_ctr_shared,
            portion=ftp_portion
        )

        # Create an instance of capped
        self.cost_recovery = CR(
            oil_capped=is_capped["oil"],
            oil_costrec=cr["oil"],
            gas_capped=is_capped["gas"],
            gas_costrec=cr["gas"]
        )

        # Create an instance of incentive
        self.incentive = Incentive(
            vat_categ=vat_categ,
            vat_percent=vat_percent,
            vat_discount=vat_discount,
            lbt_discount=lbt_discount,
            pdrd=pdrd,
            pdrd_discount=pdrd_discount,
            pdri=pdri,
            pdri_discount=pdri_discount,
            import_duty=import_duty,
            vat_import=vat_import,
            pph_import=pph_import,
            depre_accel=depre_accel,
            depre_method=depre_method
        )


@dataclass
class GrossSplitData:
    """
    Prepare PSC Gross Split data input.

    Parameters
    ----------
    field_status: dict
        Status of the field.
    field_loc: dict
        Field location.
    res_depth: dict
        Reservoir depth.
    infra_avail: dict
        Infrastructure availability.
    res_type: dict
        Reservoir type.
    api_oil: dict
        API oil.
    domestic_use: dict
        Domestic use (tingkat komponen dalam negeri).
    prod_stage: dict
        Production stage.
    co2_content: dict
        CO2 content.
    h2s_content: dict
        H2S content.
    total_split: float
        Total split.
    base_split: dict
        Base split.
    tax_rate: float
        Corporate income tax.
    branch_profit: float
        Branch profit tax.
    effective_rate: float
        Effective tax rate.
    dmo_holiday: dict
        DMO holiday (for oil and gas).
    dmo_period: dict
        DMO period (for oil and gas).
    dmo_start_production: dict
        DMO start production (for oil and gas).
    dmo_volume: dict
        DMO volume (for oil and gas).
    dmo_fee: dict
        DMO fee (for oil and gas).
    ic_available: dict
        Availability of investment credit (for oil and gas).
    ic_value: dict
        The value of investment credit (for oil and gas).
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

    # Properties used only for initialization (not assigned as class attributes)
    field_status: InitVar[dict] = field(repr=False)
    field_loc: InitVar[dict] = field(repr=False)
    res_depth: InitVar[dict] = field(repr=False)
    infra_avail: InitVar[dict] = field(repr=False)
    res_type: InitVar[dict] = field(repr=False)
    api_oil: InitVar[dict] = field(repr=False)
    domestic_use: InitVar[dict] = field(repr=False)
    prod_stage: InitVar[dict] = field(repr=False)
    co2_content: InitVar[dict] = field(repr=False)
    h2s_content: InitVar[dict] = field(repr=False)
    total_split: InitVar[float] = field(repr=False)
    base_split: InitVar[dict] = field(repr=False)
    tax_rate: InitVar[float] = field(repr=False)
    branch_profit: InitVar[float] = field(repr=False)
    effective_rate: InitVar[float] = field(repr=False)
    dmo_holiday: InitVar[dict] = field(repr=False)
    dmo_period: InitVar[dict] = field(repr=False)
    dmo_start_production: InitVar[dict] = field(repr=False)
    dmo_volume: InitVar[dict] = field(repr=False)
    dmo_fee: InitVar[dict] = field(repr=False)
    ic_available: InitVar[dict] = field(repr=False)
    ic_value: InitVar[dict] = field(repr=False)
    vat_categ: InitVar[str] = field(repr=False)
    vat_percent: InitVar[float] = field(repr=False)
    vat_discount: InitVar[float] = field(repr=False)
    lbt_discount: InitVar[float] = field(repr=False)
    pdrd: InitVar[float] = field(repr=False)
    pdrd_discount: InitVar[float] = field(repr=False)
    pdri: InitVar[float] = field(repr=False)
    pdri_discount: InitVar[float] = field(repr=False)
    import_duty: InitVar[float] = field(repr=False)
    vat_import: InitVar[float] = field(repr=False)
    pph_import: InitVar[float] = field(repr=False)
    depre_accel: InitVar[str] = field(repr=False)
    depre_method: InitVar[str] = field(repr=False)

    # The assigned attributes of the class instance
    variable_split: VariableSplit = field(default=None, init=False)
    split_config: SplitConfig = field(default=None, init=False)
    tax: Tax = field(default=None, init=False)
    dmo_oil: DMO = field(default=None, init=False)
    dmo_gas: DMO = field(default=None, init=False)
    ic: InvestmentCredit = field(default=None, init=False)
    incentive: Incentive = field(default=None, init=False)

    def __post_init__(self,
                      field_status,
                      field_loc,
                      res_depth,
                      infra_avail,
                      res_type,
                      api_oil,
                      domestic_use,
                      prod_stage,
                      co2_content,
                      h2s_content,
                      total_split,
                      base_split,
                      tax_rate,
                      branch_profit,
                      effective_rate,
                      dmo_holiday,
                      dmo_period,
                      dmo_start_production,
                      dmo_volume,
                      dmo_fee,
                      ic_available,
                      ic_value,
                      vat_categ,
                      vat_percent,
                      vat_discount,
                      lbt_discount,
                      pdrd,
                      pdrd_discount,
                      pdri,
                      pdri_discount,
                      import_duty,
                      vat_import,
                      pph_import,
                      depre_accel,
                      depre_method
                      ):
        # Create an instance of variable split
        self.variable_split = VariableSplit(
            field_status=VarSplitSet(param=field_status["param"], val=field_status["val"]),
            field_loc=VarSplitSet(param=field_loc["param"], val=field_loc["val"]),
            res_depth=VarSplitSet(param=res_depth["param"], val=res_depth["val"]),
            infra_avail=VarSplitSet(param=infra_avail["param"], val=infra_avail["val"]),
            res_type=VarSplitSet(param=res_type["param"], val=res_type["val"]),
            api_oil=VarSplitSet(param=api_oil["param"], val=api_oil["val"]),
            domestic_use=VarSplitSet(param=domestic_use["param"], val=domestic_use["val"]),
            prod_stage=VarSplitSet(param=prod_stage["param"], val=prod_stage["val"]),
            co2_content=VarSplitSet(param=co2_content["param"], val=co2_content["val"]),
            h2s_content=VarSplitSet(param=h2s_content["param"], val=h2s_content["val"]),
            total_split=total_split
        )

        # Create an instance of split config
        self.split_config = SplitConfig(
            ctr_oil=base_split["ctr_oil"],
            ctr_gas=base_split["ctr_gas"],
            ministry_discr=base_split["ministry"]
        )

        # Create an instance of tax
        self.tax = Tax(
            corporate_income=tax_rate,
            branch_profit=branch_profit,
            effective_rate=effective_rate
        )

        # Create an instance of dmo oil
        self.dmo_oil = DMO(
            holiday=dmo_holiday["oil"],
            period=dmo_period["oil"],
            start_production=dmo_start_production["oil"],
            volume=dmo_volume["oil"],
            fee=dmo_fee["oil"]
        )

        # Create an instance of dmo gas
        self.dmo_gas = DMO(
            holiday=dmo_holiday["gas"],
            period=dmo_period["gas"],
            start_production=dmo_start_production["gas"],
            volume=dmo_volume["gas"],
            fee=dmo_fee["gas"]
        )

        # Create an instance of investment credit
        self.ic = InvestmentCredit(
            oil_is_available=ic_available["oil"],
            oil_ic=ic_value["oil"],
            gas_is_available=ic_available["gas"],
            gas_ic=ic_value["gas"]
        )

        # Create an instance of incentive variable
        self.incentive = Incentive(
            vat_categ=vat_categ,
            vat_percent=vat_percent,
            vat_discount=vat_discount,
            lbt_discount=lbt_discount,
            pdrd=pdrd,
            pdrd_discount=pdrd_discount,
            pdri=pdri,
            pdri_discount=pdri_discount,
            import_duty=import_duty,
            vat_import=vat_import,
            pph_import=pph_import,
            depre_accel=depre_accel,
            depre_method=depre_method
        )


@dataclass
class CostData:
    """
    Prepare cost data input.

    Parameters
    ----------
    start_year: int
        The start year of the project.
    end_year: int
        The end year of the project.
    cost_data: np.ndarray
        Cost data input loaded from the target Excel workbook.
    """

    start_year: int
    end_year: int
    cost_data: np.ndarray = field(repr=False)

    # Attributes to be defined later in the __post_init__
    project_duration: int = field(init=False)

    def __post_init__(self):

        # Instantiate project duration and project years
        if self.start_year > self.end_year:
            raise ValueError(f"Start year {self.start_year} is after end year {self.end_year}")

        else:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        # Instantiate parameter: fluid type
        self.fluid_type = ["Oil", "Gas"]

        # Instantiate parameter: cost category
        self.cost_category = [
            "Sunk Cost",
            "Drilling Tangible",
            "Drilling Intangible",
            "Facility Tangible",
            "Operating Cost",
            "ASR",
            "Land and Building Tax",
            "PDRD",
            "Intangible Asset",
        ]

        # Instantiate parameters to decompose cost data based on their fluid (oil or gas)
        self.id_fluid_type = [0 for _ in range(len(self.fluid_type))]
        self.cost_by_fluid_type = {}

        # Instantiate parameters to decompose cost data based on their cost components
        self.id_component = {
            self.fluid_type[0]: {},
            self.fluid_type[1]: {}
        }

        self.target_component = {
            self.fluid_type[0]: {},
            self.fluid_type[1]: {}
        }

    def _arrange_cost_by_fluid_type(self) -> dict:
        """
        Decompose the cost data based on fluid type: oil or gas.

        Returns
        -------
        cost_by_fluid_type: dict
            Cost data that has been arranged based on fluid type.
        """

        # Decompose cost data based on the corresponding fluid type (oil or gas)
        # i --> iteration over the number of fluid type (i.e., oil and gas)
        for i in range(len(self.fluid_type)):
            # Row index where the fluid type matches
            self.id_fluid_type[i] = np.argwhere(self.cost_data[:, 1] == self.fluid_type[i]).ravel()

            # Fill parameter 'cost_by_fluid_type' for each corresponding fluid type
            self.cost_by_fluid_type[self.fluid_type[i]] = self.cost_data[self.id_fluid_type[i], :]

            # Delete columns #1 and #2 of array 'cost_by_fluid_type'
            self.cost_by_fluid_type[self.fluid_type[i]] = np.delete(
                self.cost_by_fluid_type[self.fluid_type[i]], obj=[1, 2], axis=1
            )

    def _arrange_target_component(self) -> dict:
        """
        Decompose cost data of each fluid type into their associated cost components, i.e.,
        sunk cost, drilling tangible, drilling intangible, facility tangible, operating cost,
        asr, land and building tax, pdrd, and intangible asset.

        Returns
        -------
        target_component: dict
            Cost data that has been arranged into their own respective components.
        """

        self._arrange_cost_by_fluid_type()

        # i --> iteration over the number of fluid types (oil and gas);
        # j --> iteration over the number of cost components
        for i in range(len(self.fluid_type)):
            for j in range(len(self.cost_category)):
                # Row index where the cost component matches
                self.id_component[self.fluid_type[i]][self.cost_category[j]] = np.argwhere(
                    self.cost_by_fluid_type[self.fluid_type[i]][:, 0] == self.cost_category[j]
                ).ravel()

                # Fill parameter: 'target_component'
                self.target_component[self.fluid_type[i]][self.cost_category[j]] = \
                    self.cost_by_fluid_type[self.fluid_type[i]][
                    self.id_component[self.fluid_type[i]][self.cost_category[j]],
                    :
                    ]

                # Delete column #0 of array target_component
                self.target_component[self.fluid_type[i]][self.cost_category[j]] = np.delete(
                    self.target_component[self.fluid_type[i]][self.cost_category[j]],
                    obj=0,
                    axis=1
                )

    @property
    def oil(self):
        """
        Generate cost components for oil.

        Returns
        -------
        oil: CostComponent
            The associated cost components of oil as an instance of CostComponent class.
        """

        self._arrange_target_component()

        return CostComponent(
            project_years=self.project_years,
            target_cost=self.target_component["Oil"]
        )

    @property
    def gas(self):
        """
        Generate cost components for gas.

        Returns
        -------
        gas: CostComponent
            The associated cost components of gas as an instance of CostComponent class.
        """

        self._arrange_target_component()

        return CostComponent(
            project_years=self.project_years,
            target_cost=self.target_component["Gas"]
        )


@dataclass
class LiftingData:
    """
    Prepare lifting data input.

    Parameters
    ----------
    start_year: int
        The start year of the project.
    end_year: int
        The end year of the project.
    prod_rate: dict
        The production rate of a particular fluid type.
    price: dict
        The associated price of a particular fluid type.
    ghv: dict
        The value of ghv of a particular fluid type (default value = 1).
    days: dict
        The number of days used as reference to specify prod_rate.
    """

    start_year: InitVar[int] = field(default=None, repr=False)
    end_year: InitVar[int] = field(default=None, repr=False)
    prod_rate: InitVar[dict] = field(default=None, repr=False)
    price: InitVar[dict] = field(default=None, repr=False)
    ghv: InitVar[dict] = field(default=None, repr=False)
    days: InitVar[dict] = field(default=None, repr=False)

    # Attributes to be defined later on
    oil: LiftingElement = field(default=None, init=False)
    condensate: LiftingElement = field(default=None, init=False)
    gsa: dict = field(default=None, init=False)
    lpg: dict = field(default=None, init=False)
    sulfur: LiftingElement = field(default=None, init=False)
    electricity: LiftingElement = field(default=None, init=False)
    co2: LiftingElement = field(default=None, init=False)
    others: dict = field(default=None, init=False)

    def __post_init__(self, start_year, end_year, prod_rate, price, ghv, days):
        # Instantiate attribute "oil"
        self.oil = LiftingElement(
            start_year=start_year,
            end_year=end_year,
            prod_rate=prod_rate["oil"],
            price=price["oil"],
            ghv=ghv["oil"],
            days=days["oil"]
        )

        # Instantiate attribute "condensate"
        self.condensate = LiftingElement(
            start_year=start_year,
            end_year=end_year,
            prod_rate=prod_rate["con"],
            price=price["con"],
            ghv=ghv["con"],
            days=days["con"]
        )

        # Instantiate attribute "gsa"
        self.gsa = {
            "gsa1": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa1"],
                price=price["gsa"]["gsa1"],
                ghv=ghv["gsa"]["gsa1"],
                days=days["gsa"]["gsa1"]
            ),
            "gsa2": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa2"],
                price=price["gsa"]["gsa2"],
                ghv=ghv["gsa"]["gsa2"],
                days=days["gsa"]["gsa2"]
            ),
            "gsa3": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa3"],
                price=price["gsa"]["gsa3"],
                ghv=ghv["gsa"]["gsa3"],
                days=days["gsa"]["gsa3"]
            ),
            "gsa4": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa4"],
                price=price["gsa"]["gsa4"],
                ghv=ghv["gsa"]["gsa4"],
                days=days["gsa"]["gsa4"]
            ),
            "gsa5": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa5"],
                price=price["gsa"]["gsa5"],
                ghv=ghv["gsa"]["gsa5"],
                days=days["gsa"]["gsa5"]
            ),
            "gsa6": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa6"],
                price=price["gsa"]["gsa6"],
                ghv=ghv["gsa"]["gsa6"],
                days=days["gsa"]["gsa6"]
            ),
            "gsa7": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa7"],
                price=price["gsa"]["gsa7"],
                ghv=ghv["gsa"]["gsa7"],
                days=days["gsa"]["gsa7"]
            ),
            "gsa8": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa8"],
                price=price["gsa"]["gsa8"],
                ghv=ghv["gsa"]["gsa8"],
                days=days["gsa"]["gsa8"]
            ),
            "gsa9": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa9"],
                price=price["gsa"]["gsa9"],
                ghv=ghv["gsa"]["gsa9"],
                days=days["gsa"]["gsa9"]
            ),
            "gsa10": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa10"],
                price=price["gsa"]["gsa10"],
                ghv=ghv["gsa"]["gsa10"],
                days=days["gsa"]["gsa10"]
            ),
            "gsa11": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa11"],
                price=price["gsa"]["gsa11"],
                ghv=ghv["gsa"]["gsa11"],
                days=days["gsa"]["gsa11"]
            ),
            "gsa12": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa12"],
                price=price["gsa"]["gsa12"],
                ghv=ghv["gsa"]["gsa12"],
                days=days["gsa"]["gsa12"]
            ),
            "gsa13": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa13"],
                price=price["gsa"]["gsa13"],
                ghv=ghv["gsa"]["gsa13"],
                days=days["gsa"]["gsa13"]
            ),
            "gsa14": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa14"],
                price=price["gsa"]["gsa14"],
                ghv=ghv["gsa"]["gsa14"],
                days=days["gsa"]["gsa14"]
            ),
            "gsa15": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa15"],
                price=price["gsa"]["gsa15"],
                ghv=ghv["gsa"]["gsa15"],
                days=days["gsa"]["gsa15"]
            ),
            "gsa16": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa16"],
                price=price["gsa"]["gsa16"],
                ghv=ghv["gsa"]["gsa16"],
                days=days["gsa"]["gsa16"]
            ),
            "gsa17": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa17"],
                price=price["gsa"]["gsa17"],
                ghv=ghv["gsa"]["gsa17"],
                days=days["gsa"]["gsa17"]
            ),
            "gsa18": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa18"],
                price=price["gsa"]["gsa18"],
                ghv=ghv["gsa"]["gsa18"],
                days=days["gsa"]["gsa18"]
            ),
            "gsa19": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa19"],
                price=price["gsa"]["gsa19"],
                ghv=ghv["gsa"]["gsa19"],
                days=days["gsa"]["gsa19"]
            ),
            "gsa20": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["gsa"]["gsa20"],
                price=price["gsa"]["gsa20"],
                ghv=ghv["gsa"]["gsa20"],
                days=days["gsa"]["gsa20"]
            )
        }

        # Instantiate attribute "lpg"
        self.lpg = {
            "propane": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["lpg"]["propane"],
                price=price["lpg"]["propane"],
                ghv=ghv["lpg"]["propane"],
                days=days["lpg"]["propane"]
            ),
            "butane": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["lpg"]["butane"],
                price=price["lpg"]["butane"],
                ghv=ghv["lpg"]["butane"],
                days=days["lpg"]["butane"]
            )
        }

        # Instantiate attribute "sulfur"
        self.sulfur = LiftingElement(
            start_year=start_year,
            end_year=end_year,
            prod_rate=prod_rate["sulfur"],
            price=price["sulfur"],
            ghv=ghv["sulfur"],
            days=days["sulfur"]
        )

        # Instantiate attribute "electricity"
        self.electricity = LiftingElement(
            start_year=start_year,
            end_year=end_year,
            prod_rate=prod_rate["electricity"],
            price=price["electricity"],
            ghv=ghv["electricity"],
            days=days["electricity"]
        )

        # Instantiate attribute "co2"
        self.co2 = LiftingElement(
            start_year=start_year,
            end_year=end_year,
            prod_rate=prod_rate["co2"],
            price=price["co2"],
            ghv=ghv["co2"],
            days=days["co2"]
        )

        # Instantiate attribute "others"
        self.others = {
            "other1": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["others"]["other1"],
                price=price["others"]["other1"],
                ghv=ghv["others"]["other1"],
                days=days["others"]["other1"]
            ),
            "other2": LiftingElement(
                start_year=start_year,
                end_year=end_year,
                prod_rate=prod_rate["others"]["other2"],
                price=price["others"]["other2"],
                ghv=ghv["others"]["other2"],
                days=days["others"]["other2"]
            )
        }
