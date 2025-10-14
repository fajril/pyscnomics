import numpy as np
from pyscnomics.api.adapter import (
    get_baseproject,
    get_costrecovery,
    get_grosssplit,
    get_transition,
)
from pyscnomics.tools.table import get_table


class Summaries:
    ctrType: int
    dataJson: dict

    def __init__(self, ctrType: int, data: dict):
        self.ctrType = ctrType
        self.dataJson = data
        summaries = (
            get_costrecovery(data=self.dataJson)
            if ctrType == 1
            else (
                get_grosssplit(data=self.dataJson)
                if ctrType == 2
                else (
                    get_transition(data=self.dataJson)
                    if ctrType >= 3
                    else get_baseproject(data=self.dataJson)
                )
            )
        )
        self.summary = summaries[0]
        # self.summary["indirect_taxes"] = get_indirect_taxes(data=self.dataJson)
        self.contract = summaries[1]

        self.tangible = (
            self.contract._oil_capital_expenditures_post_tax
            + self.contract._gas_capital_expenditures_post_tax
        ).tolist()

        self.Year = (
            self.contract.project_years
            if ctrType < 3
            else self.contract._contract1_transitioned.project_years
        ).tolist()

    def getOil(self):
        prodRate = (
            self.contract._oil_lifting.get_lifting_rate_arr()
            if self.ctrType < 3
            else (
                self.contract._contract1_transitioned._oil_lifting.get_lifting_rate_arr()
                + self.contract._contract2_transitioned._oil_lifting.get_lifting_rate_arr()
            )
        )
        return {
            "table": [prodRate.tolist(), np.cumsum(prodRate).tolist()],
            "sum": np.sum(prodRate),
        }

    def getGas(self):
        prodRate = (
            self.contract._gas_lifting.get_lifting_rate_arr()
            if self.ctrType < 3
            else (
                self.contract._contract1_transitioned._gas_lifting.get_lifting_rate_arr()
                + self.contract._contract2_transitioned._gas_lifting.get_lifting_rate_arr()
            )
        )
        return {
            "table": [prodRate.tolist(), np.cumsum(prodRate).tolist()],
            "sum": np.sum(prodRate),
        }

    def getRevenue(self):
        revenue = (
            (self.contract._oil_revenue + self.contract._gas_revenue)
            if self.ctrType == 0
            else (
                self.contract._consolidated_revenue
                if self.ctrType < 3
                else (
                    self.contract._contract1_transitioned._consolidated_revenue
                    + self.contract._contract2_transitioned._consolidated_revenue
                )
            )
        )
        return {
            "table": [revenue.tolist(), np.cumsum(revenue).tolist()],
            "sum": np.sum(revenue),
        }

    def getInvesment(self):
        return {
            "table": [self.tangible, np.cumsum(self.tangible).tolist()],
            "sum": np.sum(self.tangible),
        }

    def getExpenses(self):
        opex = (
            self.contract._oil_opex_expenditures_post_tax
            + self.contract._gas_opex_expenditures_post_tax
            if self.ctrType == 0
            else (
                self.contract._consolidated_opex
                if self.ctrType < 3
                else (
                    self.contract._contract1_transitioned._consolidated_opex
                    + self.contract._contract2_transitioned._consolidated_opex
                )
            )
        )
        return {
            "table": [
                [self.tangible, np.cumsum(self.tangible).tolist()],
                [opex.tolist(), np.cumsum(opex).tolist()],
            ],
            "sum": [np.sum(self.tangible), np.sum(opex)],
        }

    def getTax(self):
        tax = (
            None
            if self.ctrType == 0
            else (
                self.contract._consolidated_tax_payment
                if self.ctrType < 3
                else (
                    self.contract._contract1_transitioned._consolidated_tax_payment
                    + self.contract._contract2_transitioned._consolidated_tax_payment
                )
            )
        )
        return {
            "table": (
                [[0], [0]]
                if self.ctrType == 0
                else [tax.tolist(), np.cumsum(tax).tolist()]
            ),
            "sum": 0 if self.ctrType == 0 else np.sum(tax),
        }

    def getGoI(self):
        GoI = (
            None
            if self.ctrType == 0
            else (
                self.contract._consolidated_government_take
                if self.ctrType < 3
                else (
                    self.contract._contract1_transitioned._consolidated_government_take
                    + self.contract._contract2_transitioned._consolidated_government_take
                )
            )
        )
        return {
            "table": (
                [[0], [0]]
                if self.ctrType == 0
                else [GoI.tolist(), np.cumsum(GoI).tolist()]
            ),
            "sum": 0 if self.ctrType == 0 else np.sum(GoI),
        }

    def getCashFlow(self):
        cf = (
            self.contract._consolidated_cashflow
            if self.ctrType < 3
            else (
                self.contract._contract1_transitioned._consolidated_cashflow
                + self.contract._contract2_transitioned._consolidated_cashflow
            )
        )
        return {
            "table": [cf.tolist(), np.cumsum(cf).tolist()],
            "sum": np.sum(cf),
        }

    def getPie(self):
        sumAll = [0]

        def get_GoS(sum: list):
            values = (
                None
                if self.ctrType == 0
                else (
                    (
                        self.contract._consolidated_government_take
                        - self.contract._consolidated_tax_payment
                        - self.contract._consolidated_ddmo
                    )
                    if self.ctrType < 3
                    else (
                        (
                            self.contract._contract1_transitioned._consolidated_government_take
                            + self.contract._contract2_transitioned._consolidated_government_take
                        )
                        - (
                            self.contract._contract1_transitioned._consolidated_tax_payment
                            + self.contract._contract2_transitioned._consolidated_tax_payment
                        )
                        - (
                            self.contract._contract1_transitioned._consolidated_ddmo
                            + self.contract._contract2_transitioned._consolidated_ddmo
                        )
                    )
                )
            )
            sumVal = 0 if values is None else np.sum(values)
            sum[0] += sumVal
            return {
                "table": [0] if values is None else values.tolist(),
                "value": sumVal,
            }

        def get_NCS(sum: list):
            values = (
                (self.contract._oil_revenue + self.contract._gas_revenue)
                if self.ctrType == 0
                else (
                    self.contract._consolidated_ctr_net_share
                    if self.ctrType < 3
                    else (
                        self.contract._contract1_transitioned._consolidated_ctr_net_share
                        + self.contract._contract2_transitioned._consolidated_ctr_net_share
                    )
                )
            )
            sumVal = np.sum(values)
            sum[0] += sumVal
            return {"table": values.tolist(), "value": sumVal}

        def get_CR(sum: list):
            values = (
                None
                if self.ctrType == 0
                else (
                    self.contract._consolidated_cost_recovery_after_tf
                    if self.ctrType == 1
                    else (
                        self.contract._consolidated_deductible_cost
                        if self.ctrType == 2
                        else (
                            self.contract._oil_deductible_cost
                            + self.contract._gas_deductible_cost
                        )
                    )
                )
            )
            sumVal = 0 if values is None else np.sum(values)
            sum[0] += sumVal
            return {
                "table": [0] if values is None else values.tolist(),
                "value": sumVal,
            }

        def get_DMO(sum: list):
            values = (
                None
                if self.ctrType == 0
                else (
                    self.contract._consolidated_ddmo
                    if self.ctrType < 3
                    else (
                        self.contract._contract1_transitioned._consolidated_ddmo
                        + self.contract._contract2_transitioned._consolidated_ddmo
                    )
                )
            )
            sumVal = 0 if values is None else np.sum(values)
            sum[0] += sumVal
            return {
                "table": [0] if values is None else values.tolist(),
                "value": sumVal,
            }

        def get_Tax(sum: list):
            values = (
                None
                if self.ctrType == 0
                else (
                    self.contract._consolidated_tax_payment
                    if self.ctrType < 3
                    else (
                        self.contract._contract1_transitioned._consolidated_tax_payment
                        + self.contract._contract2_transitioned._consolidated_tax_payment
                    )
                )
            )
            sumVal = 0 if values is None else np.sum(values)
            sum[0] += sumVal
            return {
                "table": [0] if values is None else values.tolist(),
                "value": sumVal,
            }

        return {
            "data": {
                "GoS": get_GoS(sumAll),
                "NCS": get_NCS(sumAll),
                "CR": get_CR(sumAll),
                "DMO": get_DMO(sumAll),
                "TAX": get_Tax(sumAll),
            },
            "sum": sumAll[0],
        }
