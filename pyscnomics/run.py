"""
Execute calculations
"""




if __name__ == "__main__":

    kwargs_execute = {
        "case": Case02,
        "contract_type": ContractType.GROSS_SPLIT,
        "run_as_dict": True,
    }

    ctr = execute_contract(**kwargs_execute)

    data: dict = ctr["data"]
    contract = ctr["contract"]
    contract_arguments: dict = ctr["contract_arguments"]
    summary_arguments: dict = ctr["summary_arguments"]
    summary: dict = ctr["summary"]
    cashflow_table: pd.DataFrame = get_table(contract=contract)[0]

    # with resources.open_text("pyscnomics.dataset", "duri_field.json") as f:
    #     data = json.load(f)
    #
    # kwargs_execute = {
    #     "case": data,
    #     "contract_type": ContractType.GROSS_SPLIT,
    # }
    #
    # ctr = execute_json(**kwargs_execute)
    #
    # data: dict = ctr["data"]
    # contract = ctr["contract"]
    # contract_arguments: dict = ctr["contract_arguments"]
    # summary_arguments: dict = ctr["summary_arguments"]
    # summary: dict = ctr["summary"]
    # cashflow_table: pd.DataFrame = get_table(contract=contract)[0]

    t1 = cashflow_table
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)

    # # Run case as class's instance
    # contract = data.as_class()
    # contract_arguments = data.contract_arguments
    # summary_arguments = data.summary_arguments
    #
    # contract.run(**contract_arguments)
    # results = contract.get_summary(**summary_arguments)

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)
