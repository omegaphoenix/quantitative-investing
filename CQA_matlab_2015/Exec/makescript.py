import sys
exec_file = sys.argv[1]

with open(exec_file) as f:
    lines=f.readlines()

header="""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head profile="http://selenium-ide.openqa.org/profiles/test-case">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<link rel="selenium.base" href="http://stocktrak.com/" />
<title>script1</title>
</head>
<body>
<table cellpadding="1" cellspacing="1" border="1">
<thead>
<tr><td rowspan="1" colspan="3">script1</td></tr>
</thead><tbody>
<tr>
	<td>open</td>
	<td>/private/trading/equities.aspx</td>
	<td></td>
</tr>
"""

tail="""
</tbody></table>
</body>
</html>
"""
item = """
<tr>
    <td>select</td>
    <td>id=ContentPlaceHolder1_Equities_ddlOrderSides</td>
    <td>label={ordertype}</td>
</tr>
<tr>
	<td>type</td>
	<td>id=ContentPlaceHolder1_Equities_tbSymbol</td>
	<td>{ticker}</td>
</tr>
<tr>
	<td>pause</td>
	<td>{wait0}</td>
	<td></td>
</tr>
<tr>
	<td>type</td>
	<td>id=ContentPlaceHolder1_Equities_tbQuantity</td>
	<td>{amount}</td>
</tr>
<tr>
	<td>pause</td>
	<td>{wait0}</td>
	<td></td>
</tr>
<tr>
	<td>click</td>
	<td>id=ContentPlaceHolder1_Equities_btnPreviewOrder</td>
	<td></td>
</tr>
<tr>
	<td>pause</td>
	<td>{wait0}</td>
	<td></td>
</tr>
<tr>
	<td>click</td>
	<td>id=ContentPlaceHolder1_Equities_btnPlaceOrder</td>
	<td></td>
</tr>
<tr>
	<td>pause</td>
	<td>{wait}</td>
	<td></td>
</tr>
<tr>
	<td>click</td>
	<td>id=ContentPlaceHolder1_Equities_btnNewOrder</td>
	<td></td>
</tr>
<tr>
	<td>pause</td>
	<td>{wait0}</td>
	<td></td>
</tr>
"""
with open("script_"+exec_file,"w") as f:
    f.write(header)
    wait = 5000
    wait0 = 500
    for line in lines:
        ordertype=line.split()[0]
        ticker=line.split()[1]
        amount=line.split()[2]
        f.write(item.format(**vars()))
    f.write(tail)
