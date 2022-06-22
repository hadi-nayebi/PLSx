#!/usr/bin/env python
# coding: utf-8


from Plsx.utils.custom_argparser import TrainSessionArgParser


class TrainSession:

    pass


def main(args):
    """Run main."""
    print(args)


if __name__ == "__main__":
    parser = TrainSessionArgParser()
    parsed_args = parser.parsed()
    main(parsed_args)
