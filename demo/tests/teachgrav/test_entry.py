from teachgrav.entry import parse_args

def test_parse_args():
    args = parse_args('--scenario scatter --method rk4')
    assert args.scenario == 'scatter'
    assert args.method == 'rk4'

def test_default_args():
    args = parse_args(" ")
    assert args.scenario == 'moon'
    assert args.method == 'euler'