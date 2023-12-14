"""
Python Script as the entry point of Excel Workbook
"""
import xlwings as xw


def main(workbook_path, mode):
    # Defining the workbook object
    ws = xw.Book(workbook_path).sheets('Cover')

    if mode == 'Standard':
        result = 'Standard Result'
    else:
        result = mode

    # Test Write
    ws.range("C22").value = 'Test Run Python from Excel ' + result

    # Giving the workbook execution status
    ws.range("K15").value = 'Success'


if __name__ == '__main__':
    import sys

    main(workbook_path=sys.argv[1], mode=sys.argv[2])
    #
    # workbook_path = 'Test Entry Point.xlsb'
    # run_mode = 'Standard'
    #
    # main(workbook_path=workbook_path, mode=run_mode)
