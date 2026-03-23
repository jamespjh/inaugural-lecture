from teachgrav.entry import parse_args


def test_parse_args():
    args = parse_args(
        '--scenario scatter --method RK45 --outfile output.mp4 ' +
        '--visualise dot --video')
    assert args.scenario == 'scatter'
    assert args.method == 'RK45'
    assert args.outfile == 'output.mp4'
    assert args.visualise == 'dot'
    assert args.video
    assert args.format == 'mp4'


def test_default_args():
    args = parse_args(" ")
    assert args.scenario == 'moon'
    assert args.method == 'euler'
    assert args.outfile is None
    assert args.visualise is None
    assert not args.video
    assert args.format == 'csv'
