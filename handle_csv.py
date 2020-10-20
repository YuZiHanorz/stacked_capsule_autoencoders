import sys
import csv

infile = sys.argv[1]
outfile = sys.argv[2]


def toNum(str):
    numeric = '0123456789-.'
    str.strip()
    notUnit = True
    for i, c in enumerate(str):
        if c not in numeric:
            notUnit = False
            break
    if notUnit:
        return float(str), ''
    number = str[:i]
    unit = str[i:].lstrip()
    print(number, unit)
    return float(number), unit


with open(infile) as fin, open(outfile, mode='w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    for i in range(4):
        header_row = next(reader)
        print(header_row)
    flag = 0
    lastrow = header_row
    for rowlist in reader:
        if flag == 0:
            lastrow = rowlist
            flag = 1
            continue
        lastrowstr = ','.join(lastrow)
        rowlist = rowlist[0].split()
        rowlist.insert(0, lastrowstr)

        minN, minU = toNum(rowlist[7])
        maxN, maxU = toNum(rowlist[8])
        avgN, avgU = toNum(rowlist[9])

        if minU == '':
            print('here is transaction')
            print(rowlist)
            writer.writerow(rowlist)
            flag = True
            continue

        if minU == 'B/s':
            minN /= 1000000
            print('here B')
        if minU == 'MB/s':
            minN /= 1000
            print('here MB')
        rowlist[7] = str(minN)

        if maxU == 'B/s':
            maxN /= 1000000
        if maxU == 'MB/s':
            maxN /= 1000
        rowlist[8] = str(maxN)

        if avgU == 'B/s':
            avgN /= 1000000
        if avgU == 'MB/s':
            avgN /= 1000
        rowlist[9] = str(avgN)

        print(rowlist)
        writer.writerow(rowlist)
        flag += 1
        if flag == 3:
            flag = 0
