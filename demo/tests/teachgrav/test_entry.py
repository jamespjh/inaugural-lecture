import pytest

from teachgrav.entry import parse_args


def test_parse_args():
    args = parse_args(
        '--scenario scatter --method Tsit5 --outfile output.mp4 ' +
        '--visualise dot --video')
    assert args.scenario == 'scatter'
    assert args.method == 'Tsit5'
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
    assert args.duration == 30
    assert args.format == 'csv'


def test_duration_with_video_args():
    args = parse_args('--video --duration 45 --outfile output.mp4')
    assert args.video
    assert args.duration == 45


def test_duration_without_video_raises():
    with pytest.raises(
            ValueError,
            match='Option --duration can only be used with video output'):
        parse_args('--duration 45')
