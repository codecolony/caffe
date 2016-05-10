import io
import csv
output = io.StringIO()
csvdata = ["'filename, xmin, ymin, xmax, ymax'\n'group.jpg', 50, 50, 200, 200"]
writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
writer.writerow(csvdata)

Some details need to be changed a bit for Python 2:

# >>> output = io.BytesIO()
# >>> writer = csv.writer(output)
# >>> writer.writerow(csvdata)
# 57L
# >>> output.getvalue()
# '1,2,a,"He said ""what do you mean?""","Whoa!\nNewlines!"\r\n'

