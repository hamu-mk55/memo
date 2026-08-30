from line_profiler import LineProfiler

from anomaly_detector import main

def exam_func1(input=10):
    _ret = exam_func2(input, repeat_num=10000)

    return _ret


def exam_func2(input=10, repeat_num=1000):
    for _ in range(repeat_num):
        _ret = input + 10
    return _ret


def check_profiler():
    """
    プロファイリングを実行
    :return:
    """

    prof = LineProfiler()

    # register functions
    prof.add_function(main)
    # prof.add_function(exam_func2)

    # call
    for _ in range(10):
        prof.runcall(lambda: main(debug=False))

    prof.print_stats(output_unit=1e-3)


if __name__ == '__main__':
    check_profiler()