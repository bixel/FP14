import numpy
import uncertainties


def numplaces(num, uncert=False):
    if uncert:
        l, r = '{:f}'.format(num).split('+/-')
        try:
            a, b = l.split('.')
            c, d = r.split('.')
        except:
            a = '{:.0f}'.format(round(float(l) / 10) * 10)
            b = ''
            d = '{:.0f}'.format(round(float(r) / 10) * 10)
        return len(a), len(b), len(d)
    else:
        # TODO temporary hack
        if isinstance(num, float):
            a, b = '{}'.format(num).split('.')
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


def genspec(col):
    amax = 0
    bmax = 0
    cmax = 0
    if isinstance(col, numpy.ndarray):
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
    if not isinstance(cols[0], uncertainties.AffineScalarFunc) \
            and not isinstance(cols[0], numpy.float64):
        for c in cols:
            maxlen = max(len(c), maxlen)
    else:
        for c in cols:
            maxlen = max(1, maxlen)
    for i in range(maxlen):
        for c in cols:
            try:
                if not isinstance(cols[0], uncertainties.AffineScalarFunc) \
                        and not isinstance(cols[0], numpy.float64):
                    if is_uncert(c[i]):
                        line.append('{:fL}'.format(c[i]))
                    else:
                        line.append('{}'.format(c[i]))
                else:
                    if is_uncert(c):
                        line.append('{:fL}'.format(c))
                    else:
                        line.append('{}'.format(c))

            except:
                line.append('')
        result.append(' & '.join(line) + r' \\')
        line = []

    result.append(r'\bottomrule')
    result.append(r'\end{tabular}')

    return '\n'.join(result)
