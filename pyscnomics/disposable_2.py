
def _get_oil_capital(self) -> CapitalCost:
    """
    Determines total oil CapitalCost from the number of oil CapitalCost instances in
    attribute self.capital_cost_total.

    Returns
    -------
    CapitalCost
        An instance of CapitalCost that only includes FluidType.OIL as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of CapitalCost class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.capital_cost_total,
    (2) If OIL is not available as an instance in attribute self.capital_cost_total,
        then establish a new instance of OIL CapitalCost with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.OIL in attribute
        self.capital_cost_total,
    (4) Create a new instance of CapitalCost with only FluidType.OIL as its cost_allocation.
    """

    if FluidType.OIL not in self.capital_cost_total.cost_allocation:
        return CapitalCost(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.OIL],
        )

    else:
        # Configure indices of OIL capital cost
        mask = np.logical_and(
            np.array(self.capital_cost_total.cost_allocation) == FluidType.OIL,
            np.array(self.capital_cost_total.is_sunkcost) == False
        )

        oil_capital_id = np.flatnonzero(mask)

        start_year = self.capital_cost_total.start_year
        end_year = self.capital_cost_total.end_year
        expense_year = self.capital_cost_total.expense_year[oil_capital_id]
        cost = self.capital_cost_total.cost[oil_capital_id]
        cost_allocation = np.array(self.capital_cost_total.cost_allocation)[oil_capital_id]
        description = np.array(self.capital_cost_total.description)[oil_capital_id]
        is_sunkcost = np.array(self.capital_cost_total.is_sunkcost)[oil_capital_id]
        tax_portion = self.capital_cost_total.tax_portion[oil_capital_id]
        tax_discount = self.capital_cost_total.tax_discount[oil_capital_id]
        pis_year = self.capital_cost_total.pis_year[oil_capital_id]
        salvage_value = self.capital_cost_total.salvage_value[oil_capital_id]
        useful_life = self.capital_cost_total.useful_life[oil_capital_id]
        depreciation_factor = self.capital_cost_total.depreciation_factor[oil_capital_id]
        is_ic_applied = np.array(self.capital_cost_total.is_ic_applied)[oil_capital_id]

        return CapitalCost(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            is_sunkcost=is_sunkcost.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            pis_year=pis_year,
            salvage_value=salvage_value,
            useful_life=useful_life,
            depreciation_factor=depreciation_factor,
            is_ic_applied=is_ic_applied.tolist(),
        )



def _get_gas_capital(self) -> CapitalCost:
    """
    Determines total gas CapitalCost from the number of gas CapitalCost instances in
    attribute self.capital_cost_total.

    Returns
    -------
    CapitalCost
        An instance of CapitalCost that only includes FluidType.GAS as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of CapitalCost class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.capital_cost_total,
    (2) If GAS is not available as an instance in attribute self.capital_cost_total,
        then establish a new instance of GAS CapitalCost with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.GAS in attribute
        self.capital_cost_total,
    (4) Create a new instance of CapitalCost with only FluidType.GAS as its cost_allocation.
    """

    if FluidType.GAS not in self.capital_cost_total.cost_allocation:
        return CapitalCost(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.GAS],
        )

    else:
        gas_capital_id = np.argwhere(
            np.array(self.capital_cost_total.cost_allocation) == FluidType.GAS
        ).ravel()

        start_year = self.capital_cost_total.start_year
        end_year = self.capital_cost_total.end_year
        expense_year = self.capital_cost_total.expense_year[gas_capital_id]
        cost = self.capital_cost_total.cost[gas_capital_id]
        cost_allocation = np.array(self.capital_cost_total.cost_allocation)[gas_capital_id]
        description = np.array(self.capital_cost_total.description)[gas_capital_id]
        tax_portion = self.capital_cost_total.tax_portion[gas_capital_id]
        tax_discount = self.capital_cost_total.tax_discount[gas_capital_id]
        pis_year = self.capital_cost_total.pis_year[gas_capital_id]
        salvage_value = self.capital_cost_total.salvage_value[gas_capital_id]
        useful_life = self.capital_cost_total.useful_life[gas_capital_id]
        depreciation_factor = self.capital_cost_total.depreciation_factor[gas_capital_id]
        is_ic_applied = np.array(self.capital_cost_total.is_ic_applied)[gas_capital_id]

        return CapitalCost(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            pis_year=pis_year,
            salvage_value=salvage_value,
            useful_life=useful_life,
            depreciation_factor=depreciation_factor,
            is_ic_applied=is_ic_applied.tolist(),
        )
        
        
        
def _get_oil_intangible(self) -> Intangible:
    """
    Determines total oil Intangible from the number of oil Intangible instances in
    attribute self.intangible_cost_total.

    Returns
    -------
    Intangible
        An instance of Intangible that only includes FluidType.OIL as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of Intangible class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.intangible_cost_total,
    (2) If OIL is not available as an instance in attribute self.intangible_cost_total,
        then establish a new instance of OIL Intangible with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.OIL in attribute
        self.intangible_cost_total,
    (4) Create a new instance of Intangible with only FluidType.OIL as its cost_allocation.
    """

    if FluidType.OIL not in self.intangible_cost_total.cost_allocation:
        return Intangible(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.OIL],
        )

    else:
        oil_intangible_id = np.argwhere(
            np.array(self.intangible_cost_total.cost_allocation) == FluidType.OIL
        ).ravel()

        start_year = self.intangible_cost_total.start_year
        end_year = self.intangible_cost_total.end_year
        expense_year = self.intangible_cost_total.expense_year[oil_intangible_id]
        cost = self.intangible_cost_total.cost[oil_intangible_id]
        cost_allocation = np.array(
            self.intangible_cost_total.cost_allocation
        )[oil_intangible_id]
        description = np.array(self.intangible_cost_total.description)[oil_intangible_id]
        tax_portion = self.intangible_cost_total.tax_portion[oil_intangible_id]
        tax_discount = self.intangible_cost_total.tax_discount[oil_intangible_id]

        return Intangible(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
        )
            
            
            
def _get_gas_intangible(self) -> Intangible:
    """
    Determines total gas Intangible from the number of gas Intangible instances in
    attribute self.intangible_cost_total.

    Returns
    -------
    Intangible
        An instance of Intangible that only includes FluidType.GAS as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of Intangible class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.intangible_cost_total,
    (2) If GAS is not available as an instance in attribute self.intangible_cost_total,
        then establish a new instance of GAS Intangible with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.GAS in attribute
        self.intangible_cost_total,
    (4) Create a new instance of Intangible with only FluidType.GAS as its cost_allocation.
    """

    if FluidType.GAS not in self.intangible_cost_total.cost_allocation:
        return Intangible(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.GAS],
        )

    else:
        gas_intangible_id = np.argwhere(
            np.array(self.intangible_cost_total.cost_allocation) == FluidType.GAS
        ).ravel()

        start_year = self.intangible_cost_total.start_year
        end_year = self.intangible_cost_total.end_year
        expense_year = self.intangible_cost_total.expense_year[gas_intangible_id]
        cost = self.intangible_cost_total.cost[gas_intangible_id]
        cost_allocation = np.array(self.intangible_cost_total.cost_allocation)[gas_intangible_id]
        description = np.array(self.intangible_cost_total.description)[gas_intangible_id]
        tax_portion = self.intangible_cost_total.tax_portion[gas_intangible_id]
        tax_discount = self.intangible_cost_total.tax_discount[gas_intangible_id]

        return Intangible(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
        )
        
        
        
def _get_oil_opex(self) -> OPEX:
    """
    Determines total oil OPEX from the number of oil OPEX instances in
    attribute self.opex_total.

    Returns
    -------
    OPEX
        An instance of OPEX that only includes FluidType.OIL as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of OPEX class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.opex_total,
    (2) If OIL is not available as an instance in attribute self.opex_total,
        then establish a new instance of OIL OPEX with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.OIL in attribute
        self.opex_total,
    (4) Create a new instance of OPEX with only FluidType.OIL as its cost_allocation.
    """

    if FluidType.OIL not in self.opex_total.cost_allocation:
        return OPEX(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            fixed_cost=np.array([0]),
            cost_allocation=[FluidType.OIL],
        )

    else:
        oil_opex_id = np.argwhere(
            np.array(self.opex_total.cost_allocation) == FluidType.OIL
        ).ravel()

        start_year = self.opex_total.start_year
        end_year = self.opex_total.end_year
        expense_year = self.opex_total.expense_year[oil_opex_id]
        cost_allocation = np.array(self.opex_total.cost_allocation)[oil_opex_id]
        description = np.array(self.opex_total.description)[oil_opex_id]
        tax_portion = self.opex_total.tax_portion[oil_opex_id]
        tax_discount = self.opex_total.tax_discount[oil_opex_id]
        fixed_cost = self.opex_total.fixed_cost[oil_opex_id]
        prod_rate = self.opex_total.prod_rate[oil_opex_id]
        cost_per_volume = self.opex_total.cost_per_volume[oil_opex_id]

        return OPEX(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            fixed_cost=fixed_cost,
            prod_rate=prod_rate,
            cost_per_volume=cost_per_volume,
        )
        
        
        
def _get_gas_opex(self) -> OPEX:
    """
    Determines total gas OPEX from the number of gas OPEX instances in
    attribute self.opex_total.

    Returns
    -------
    OPEX
        An instance of OPEX that only includes FluidType.GAS as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of OPEX class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.opex_total,
    (2) If GAS is not available as an instance in attribute self.opex_total,
        then establish a new instance of GAS OPEX with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.GAS in attribute
        self.opex_total,
    (4) Create a new instance of OPEX with only FluidType.GAS as its cost_allocation.
    """

    if FluidType.GAS not in self.opex_total.cost_allocation:
        return OPEX(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            fixed_cost=np.array([0]),
            cost_allocation=[FluidType.GAS],
        )

    else:
        gas_opex_id = np.argwhere(
            np.array(self.opex_total.cost_allocation) == FluidType.GAS
        ).ravel()

        start_year = self.opex_total.start_year
        end_year = self.opex_total.end_year
        expense_year = self.opex_total.expense_year[gas_opex_id]
        cost_allocation = np.array(self.opex_total.cost_allocation)[gas_opex_id]
        description = np.array(self.opex_total.description)[gas_opex_id]
        tax_portion = self.opex_total.tax_portion[gas_opex_id]
        tax_discount = self.opex_total.tax_discount[gas_opex_id]
        fixed_cost = self.opex_total.fixed_cost[gas_opex_id]
        prod_rate = self.opex_total.prod_rate[gas_opex_id]
        cost_per_volume = self.opex_total.cost_per_volume[gas_opex_id]

        return OPEX(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            fixed_cost=fixed_cost,
            prod_rate=prod_rate,
            cost_per_volume=cost_per_volume,
        )
        
        
        
def _get_oil_asr(self) -> ASR:
    """
    Determines total oil ASR from the number of oil ASR instances in
    attribute self.asr_cost_total.

    Returns
    -------
    ASR
        An instance of ASR that only includes FluidType.OIL as the associated
        cost_allocation that has been combined altogether following the rules
        prescribed in the dunder method __add__() of ASR class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.asr_cost_total,
    (2) If OIL is not available as an instance in attribute self.asr_cost_total,
        then establish a new instance of OIL ASR with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.OIL in attribute
        self.asr_cost_total,
    (4) Create a new instance of ASR with only FluidType.OIL as its cost_allocation.
    """

    if FluidType.OIL not in self.asr_cost_total.cost_allocation:
        return ASR(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.OIL]
        )

    else:
        oil_asr_id = np.argwhere(
            np.array(self.asr_cost_total.cost_allocation) == FluidType.OIL
        ).ravel()

        start_year = self.asr_cost_total.start_year
        end_year = self.asr_cost_total.end_year
        expense_year = self.asr_cost_total.expense_year[oil_asr_id]
        cost = self.asr_cost_total.cost[oil_asr_id]
        cost_allocation = np.array(self.asr_cost_total.cost_allocation)[oil_asr_id]
        description = np.array(self.asr_cost_total.description)[oil_asr_id]
        tax_portion = self.asr_cost_total.tax_portion[oil_asr_id]
        tax_discount = self.asr_cost_total.tax_discount[oil_asr_id]
        final_year = self.asr_cost_total.final_year[oil_asr_id]
        future_rate = self.asr_cost_total.future_rate[oil_asr_id]

        return ASR(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            final_year=final_year,
            future_rate=future_rate,
        )


    def _get_gas_asr(self) -> ASR:
        """
        Determines total gas ASR from the number of gas ASR instances in
        attribute self.asr_cost_total.

        Returns
        -------
        ASR
            An instance of ASR that only includes FluidType.GAS as the associated
            cost_allocation that has been combined altogether following the rules
            prescribed in the dunder method __add__() of ASR class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.asr_cost_total,
        (2) If GAS is not available as an instance in attribute self.asr_cost_total,
            then establish a new instance of GAS ASR with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.GAS in attribute
            self.asr_cost_total,
        (4) Create a new instance of ASR with only FluidType.GAS as its cost_allocation.
        """

        if FluidType.GAS not in self.asr_cost_total.cost_allocation:
            return ASR(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_asr_id = np.argwhere(
                np.array(self.asr_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.asr_cost_total.start_year
            end_year = self.asr_cost_total.end_year
            expense_year = self.asr_cost_total.expense_year[gas_asr_id]
            cost = self.asr_cost_total.cost[gas_asr_id]
            cost_allocation = np.array(self.asr_cost_total.cost_allocation)[gas_asr_id]
            description = np.array(self.asr_cost_total.description)[gas_asr_id]
            tax_portion = self.asr_cost_total.tax_portion[gas_asr_id]
            tax_discount = self.asr_cost_total.tax_discount[gas_asr_id]
            final_year = self.asr_cost_total.final_year[gas_asr_id]
            future_rate = self.asr_cost_total.future_rate[gas_asr_id]

            return ASR(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                final_year=final_year,
                future_rate=future_rate,
            ) 
            
            
def _get_oil_lbt(self) -> LBT:
    """
    Determines total oil LBT from the number of oil LBT instances in
    attribute self.lbt_cost_total.

    Returns
    -------
    LBT
        An instance of LBT that only includes FluidType.OIL as the associated
        cost_allocation that has been combined altogether following the rules
        prescribed in the dunder method __add__() of LBT class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.lbt_cost_total,
    (2) If OIL is not available as an instance in attribute self.lbt_cost_total,
        then establish a new instance of OIL LBT with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.OIL in attribute
        self.lbt_cost_total,
    (4) Create a new instance of LBT with only FluidType.OIL as its cost_allocation.
    """

    if FluidType.OIL not in self.lbt_cost_total.cost_allocation:
        return LBT(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost_allocation=[FluidType.OIL],
        )

    else:
        oil_lbt_id = np.argwhere(
            np.array(self.lbt_cost_total.cost_allocation) == FluidType.OIL
        ).ravel()

        start_year = self.lbt_cost_total.start_year
        end_year = self.lbt_cost_total.end_year
        expense_year = self.lbt_cost_total.expense_year[oil_lbt_id]
        cost_allocation = np.array(self.lbt_cost_total.cost_allocation)[oil_lbt_id]
        description = np.array(self.lbt_cost_total.description)[oil_lbt_id]
        tax_portion = self.lbt_cost_total.tax_portion[oil_lbt_id]
        tax_discount = self.lbt_cost_total.tax_discount[oil_lbt_id]
        final_year = self.lbt_cost_total.final_year[oil_lbt_id]
        utilized_land_area = self.lbt_cost_total.utilized_land_area[oil_lbt_id]
        utilized_building_area = self.lbt_cost_total.utilized_building_area[oil_lbt_id]
        njop_land = self.lbt_cost_total.njop_land[oil_lbt_id]
        njop_building = self.lbt_cost_total.njop_building[oil_lbt_id]
        gross_revenue = self.lbt_cost_total.gross_revenue[oil_lbt_id]
        cost = self.lbt_cost_total.cost[oil_lbt_id]

        return LBT(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            final_year=final_year,
            utilized_land_area=utilized_land_area,
            utilized_building_area=utilized_building_area,
            njop_land=njop_land,
            njop_building=njop_building,
            gross_revenue=gross_revenue,
            cost=cost,
        )     



def _get_gas_lbt(self) -> LBT:
    """
    Determines total gas LBT from the number of gas LBT instances in
    attribute self.lbt_cost_total.

    Returns
    -------
    LBT
        An instance of LBT that only includes FluidType.GAS as the associated
        cost_allocation that has been combined altogether following the rules
        prescribed in the dunder method __add__() of LBT class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.lbt_cost_total,
    (2) If GAS is not available as an instance in attribute self.lbt_cost_total,
        then establish a new instance of GAS LBT with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.GAS in attribute
        self.lbt_cost_total,
    (4) Create a new instance of LBT with only FluidType.GAS as its cost_allocation.
    """

    if FluidType.GAS not in self.lbt_cost_total.cost_allocation:
        return LBT(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=[self.start_date.year],
            cost_allocation=[FluidType.GAS]
        )

    else:
        gas_lbt_id = np.argwhere(
            np.array(self.lbt_cost_total.cost_allocation) == FluidType.GAS
        ).ravel()

        start_year = self.lbt_cost_total.start_year
        end_year = self.lbt_cost_total.end_year
        expense_year = self.lbt_cost_total.expense_year[gas_lbt_id]
        cost_allocation = np.array(self.lbt_cost_total.cost_allocation)[gas_lbt_id]
        description = np.array(self.lbt_cost_total.description)[gas_lbt_id]
        tax_portion = self.lbt_cost_total.tax_portion[gas_lbt_id]
        tax_discount = self.lbt_cost_total.tax_discount[gas_lbt_id]
        final_year = self.lbt_cost_total.final_year[gas_lbt_id]
        utilized_land_area = self.lbt_cost_total.utilized_land_area[gas_lbt_id]
        utilized_building_area = self.lbt_cost_total.utilized_building_area[gas_lbt_id]
        njop_land = self.lbt_cost_total.njop_land[gas_lbt_id]
        njop_building = self.lbt_cost_total.njop_building[gas_lbt_id]
        gross_revenue = self.lbt_cost_total.gross_revenue[gas_lbt_id]
        cost = self.lbt_cost_total.cost[gas_lbt_id]

        return LBT(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            final_year=final_year,
            utilized_land_area=utilized_land_area,
            utilized_building_area=utilized_building_area,
            njop_land=njop_land,
            njop_building=njop_building,
            gross_revenue=gross_revenue,
            cost=cost,
        ) 
        
        
        
def _get_oil_cost_of_sales(self) -> CostOfSales:
    """
    Retrieve the oil cost of sales from the total cost of sales data.

    If oil-related costs are not found in the cost allocation, returns a default
    `CostOfSales` instance with zero cost. Otherwise, extracts and returns the
    relevant cost details for oil from `cost_of_sales_total`.

    Returns
    -------
    CostOfSales
        An instance of `CostOfSales` containing the cost of sales data for oil,
        including start and end years, expense year, cost, cost allocation,
        description, tax portion, and tax discount.

    Notes
    -----
    - The method checks if `FluidType.OIL` exists in `cost_of_sales_total.cost_allocation`.
    - If it does not exist, a default `CostOfSales` with zero cost is returned.
    - If it exists, relevant attributes are extracted using `np.argwhere` and returned.
    """

    if FluidType.OIL not in self.cost_of_sales_total.cost_allocation:
        return CostOfSales(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=self.project_years,
            cost=np.zeros_like(self.project_years, dtype=np.float64),
            cost_allocation=[FluidType.OIL for _ in range(self.project_duration)],
        )

    else:
        oil_cost_of_sales_id = np.argwhere(
            np.array(self.cost_of_sales_total.cost_allocation) == FluidType.OIL
        ).ravel()

        start_year = self.cost_of_sales_total.start_year
        end_year = self.cost_of_sales_total.end_year
        expense_year = self.cost_of_sales_total.expense_year[oil_cost_of_sales_id]
        cost = self.cost_of_sales_total.cost[oil_cost_of_sales_id]
        cost_allocation = np.array(
            self.cost_of_sales_total.cost_allocation
        )[oil_cost_of_sales_id]
        description = np.array(
            self.cost_of_sales_total.description
        )[oil_cost_of_sales_id]
        tax_portion = self.cost_of_sales_total.tax_portion[oil_cost_of_sales_id]
        tax_discount = self.cost_of_sales_total.tax_discount[oil_cost_of_sales_id]

        return CostOfSales(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
        )
        
        
        
def _get_gas_cost_of_sales(self) -> CostOfSales:
    """
    Retrieve the gas cost of sales from the total cost of sales data.

    If gas-related costs are not found in the cost allocation, returns a default
    `CostOfSales` instance with zero cost. Otherwise, extracts and returns the
    relevant cost details for gas from `cost_of_sales_total`.

    Returns
    -------
    CostOfSales
        An instance of `CostOfSales` containing the cost of sales data for gas,
        including start and end years, expense year, cost, cost allocation,
        description, tax portion, and tax discount.

    Notes
    -----
    - The method checks if `FluidType.GAS` exists in `cost_of_sales_total.cost_allocation`.
    - If it does not exist, a default `CostOfSales` with zero cost is returned.
    - If it exists, relevant attributes are extracted using `np.argwhere` and returned.
    """

    if FluidType.GAS not in self.cost_of_sales_total.cost_allocation:
        return CostOfSales(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=self.project_years,
            cost=np.zeros_like(self.project_years, dtype=np.float64),
            cost_allocation=[FluidType.GAS for _ in range(self.project_duration)],
        )

    else:
        gas_cost_of_sales_id = np.argwhere(
            np.array(self.cost_of_sales_total.cost_allocation) == FluidType.GAS
        ).ravel()

        start_year = self.cost_of_sales_total.start_year
        end_year = self.cost_of_sales_total.end_year
        expense_year = self.cost_of_sales_total.expense_year[gas_cost_of_sales_id]
        cost = self.cost_of_sales_total.cost[gas_cost_of_sales_id]
        cost_allocation = np.array(
            self.cost_of_sales_total.cost_allocation
        )[gas_cost_of_sales_id]
        description = np.array(
            self.cost_of_sales_total.description
        )[gas_cost_of_sales_id]
        tax_portion = self.cost_of_sales_total.tax_portion[gas_cost_of_sales_id]
        tax_discount = self.cost_of_sales_total.tax_discount[gas_cost_of_sales_id]

        return CostOfSales(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
        )
        