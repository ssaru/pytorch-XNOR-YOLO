"""BinaryConnect: Training Deep Neural Networks with binary weights during propagations
Usage:
    main.py <command> [<args>...]
    main.py (-h | --help)
Available commands:    
    train
    predict
    evaluate
    test
Options:
    -h --help     Show this.
See 'python main.py <command> --help' for more information on a specific command.
"""

from pathlib import Path

from type_docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__, options_first=True)
    argv = [args["<command>"]] + args["<args>"]

    if args["<command>"] == "train":
        from train import __doc__, train

        train(docopt(__doc__, argv=argv, types={"path": Path}))

    elif args["<command>"] == "predict":
        from infer import __doc__, infer

        infer(docopt(__doc__, argv=argv, types={"path": Path}))
    
    elif args["<command>"] == "evaluate":
        from evaluate import __doc__, evaluate

        evaluate(docopt(__doc__, argv=argv, types={"path": Path}))

    else:
        raise NotImplementedError(f"Command does not exist: {args['<command>']}")
