import numpy
import uncertainties
from uncertainties import ufloat


def numplaces(num, uncert=False):
    if uncert:
        l, r = '{:f}'.format(num).split('+/-')
        try:
            a, b = l.split('.')
            c, d = r.split('.')
            if int(b[:2]) == 0 and int(d[:2]) == 0:
                b = ''
                d = ''
            elif int(b[:2]) == 0:
                b = ''
        except:
            a = '{:.0f}'.format(round(float(l) / 10) * 10)
            b = ''
            d = '{:.0f}'.format(round(float(r) / 10) * 10)
        return len(a), len(b), len(d)
    else:
        # TODO temporary hack
        if isinstance(num, float):
            a, b = '{}'.format(num).split('.')
            if int(b) == 0:
                b = ''
            return len(a), len(b)
        else:
            a = num
            b = 0
            return len(str(a)), len(str(b))


def is_uncert(num):
    uncert = True
    try:
        num.nominal_value
        num.std_dev
    except:
        uncert = False
    return uncert


def max_length(col):
    amax = 0
    bmax = 0
    cmax = 0
    for v in col:
        if is_uncert(v):
            a, b, c = numplaces(v, uncert=True)
            if a > amax:
                amax = a
            if b > bmax:
                bmax = b
            if c > cmax:
                cmax = c
        else:
            a, b = numplaces(v)
            if a > amax:
                amax = a
            if b > bmax:
                bmax = b
    if cmax:
        return amax, bmax, cmax
    else:
        return amax, bmax

def genspec(col):
    cmax = 0
    amax = 0
    bmax = 0
    if isinstance(col, (numpy.ndarray, list)):
        maximum = max_length(col)
        if len(maximum) == 2:
            amax = maximum[0]
            bmax = maximum[1]
        else:
            amax = maximum[0]
            bmax = maximum[1]
            cmax = maximum[2]
        if bmax > 2:
            bmax = 2
            if cmax:
                cmax = 2
        if cmax:
            return 'S[table-format={}.{}({})]'.format(amax, bmax, cmax)
        else:
            return 'S[table-format={}.{}]'.format(amax, bmax)
    else:
        v = col
        if is_uncert(v):
            a, b, c = numplaces(v, uncert=True)
            if a > amax:
                amax = a
            if b > bmax:
                bmax = b
            if c > cmax:
                cmax = c
        else:
            a, b = numplaces(v)
            if a > amax:
                amax = a
            if b > bmax:
                bmax = b
        if bmax > 2:
            bmax = 2
            if cmax:
                cmax = 2
        if cmax:
            return 'S[table-format={}.{}({})]'.format(amax, bmax, cmax)
        else:
            return 'S[table-format={}.{}]'.format(amax, bmax)


def table(names, cols):
    result = []
    spec = ' '.join(map(genspec, cols))
    result.append(r'\begin{{tabular}}{{{}}}'.format(spec))
    result.append(r'\toprule')
    result.append(' & '.join(map(r'\multicolumn{{1}}{{c}}{{{}}}'.format,
                                 names)) + r'\\')
    result.append(r'\midrule')
    line = []
    maxlen = 0
    if not isinstance(cols[0], uncertainties.core.AffineScalarFunc) \
            and not isinstance(cols[0], numpy.float64):
        for c in cols:
            maxlen = max(len(c), maxlen)
    else:
        for c in cols:
            maxlen = max(1, maxlen)
    for i in range(maxlen):
        for c in cols:
            try:
                if not isinstance(cols[0], uncertainties.core.AffineScalarFunc) \
                        and not isinstance(cols[0], numpy.float64):
                    maximum = max_length(c)
                    if maximum[1] > 2:
                        roundNumber = 2
                    else:
                        roundNumber = maximum[1]
                    if is_uncert(c[i]):
                        splitpm = '{:fL}'.format(c[i]).split(' \\pm ')
                        if len(splitpm[1].split('.')) == 1:
                            c[i] = ufloat(c[i].nominal_value, float(c[i].std_dev) + 1*1e-20)
                        splitpm = '{:fL}'.format(c[i]).split(' \\pm ')
                        splitEntry = splitpm[0].split('.')
                        if len(splitEntry[-1]) > roundNumber:
                            newNumber = '{1:{0}.{2}fL}'.format(len(splitEntry[0]), c[i], roundNumber)
                            if float(newNumber.split(' \\pm ')[-1]) == 0:
                                newNumber = newNumber.split(' \\pm ')[0]
                            line.append(newNumber)
                        else:
                            line.append('{:fL}'.format(c[i]))
                    else:
                        splitEntry = str(c[i]).split('.')
                        if len(splitEntry[-1]) > roundNumber:
                            line.append('{1:{0}.{2}f}'.format(len(splitEntry[0]), c[i], roundNumber))
                        else:
                            line.append('{}'.format(c[i]))
                else:
                    if is_uncert(c):
                        line.append('{:fL}'.format(c))
                    else:
                        line.append('{}'.format(c))
            except IndexError as e:
                line.append('')
            except Exception as e:
                if isinstance(c[i], (numpy.str_, str)):
                    line.append('{}'.format(c[i].replace('_', '\_')))
                else:
                    line.append('')
        result.append(' & '.join(line) + r' \\')
        line = []

    result.append(r'\bottomrule')
    result.append(r'\end{tabular}')

    return '\n'.join(result)
