Teaching Fasttext to play Chess
===============================

This is an experiment testing how well the simple one-layer + soft-max model of fasttext.cc
can learn to predict the best chess move and game evaluation.

Screenshot
==========

    $ python play_chess.py s105.bin
    Loading model...
    Do you want to be white or black? white

    8 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
    7 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
    6 · · · · · · · ·
    5 · · · · · · · ·
    4 · · · · · · · ·
    3 · · · · · · · ·
    2 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
    1 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
      a b c d e f g h

    Your move (e.g. g1f3 or Nf3): e4
    0: 0.19274, d2d4: 0.19013, t_d4: 0.17749, f_d2: 0.16625, -1: 0.045802, t_c3: 0.037613, b1c3: 0.028387, f_b1: 0.027638, f_e2: 0.020329, 1: 0.019594
    My move: d5 score=4cp

    8 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
    7 ♙ ♙ ♙ · ♙ ♙ ♙ ♙
    6 · · · · · · · ·
    5 · · · ♙ · · · ·
    4 · · · · ♟ · · ·
    3 · · · · · · · ·
    2 ♟ ♟ ♟ ♟ · ♟ ♟ ♟
    1 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
      a b c d e f g h

    Your move (e.g. g1f3 or Nf3): Nc3
    f_d4: 0.28902, d4d5: 0.10902, d4e5: 0.088574, t_d5: 0.073413, 0: 0.071389, t_e5: 0.06572, f_e2: 0.027048, 1: 0.026098, -1: 0.02319, t_f3: 0.022698
    My move: d4 score=-1cp

    8 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
    7 ♙ ♙ ♙ · ♙ ♙ ♙ ♙
    6 · · · · · · · ·
    5 · · · · · · · ·
    4 · · · ♙ ♟ · · ·
    3 · · ♞ · · · · ·
    2 ♟ ♟ ♟ ♟ · ♟ ♟ ♟
    1 ♜ · ♝ ♛ ♚ ♝ ♞ ♜
      a b c d e f g h

    Your move (e.g. g1f3 or Nf3): Nb5
    a2a3: 0.11108, 0: 0.072122, f_a2: 0.066892, f_c2: 0.064766, c2c3: 0.06256, t_a3: 0.062459, t_e4: 0.053224, f_e2: 0.052361, t_c3: 0.051711, 1: 0.03495
    My move: a6 score=-16cp

    8 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
    7 · ♙ ♙ · ♙ ♙ ♙ ♙
    6 ♙ · · · · · · ·
    5 · ♞ · · · · · ·
    4 · · · ♙ ♟ · · ·
    3 · · · · · · · ·
    2 ♟ ♟ ♟ ♟ · ♟ ♟ ♟
    1 ♜ · ♝ ♛ ♚ ♝ ♞ ♜
      a b c d e f g h

    Your move (e.g. g1f3 or Nf3): Nxd4
    t_d5: 0.1696, f_e2: 0.10756, t_e4: 0.079069, e2e4: 0.065545, 0: 0.060323, 1: 0.043838, t_f3: 0.04056, 2: 0.034371, g1f3: 0.02841, f_d1: 0.027859
    My move: e5 score=-65cp

    8 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
    7 · ♙ ♙ · · ♙ ♙ ♙
    6 ♙ · · · · · · ·
    5 · · · · ♙ · · ·
    4 · · · ♞ ♟ · · ·
    3 · · · · · · · ·
    2 ♟ ♟ ♟ ♟ · ♟ ♟ ♟
    1 ♜ · ♝ ♛ ♚ ♝ ♞ ♜
      a b c d e f g h

    Your move (e.g. g1f3 or Nf3):

Run it!
=======

You'll need the following libraries:

    git clone git@github.com:facebookresearch/fastText.git
    git clone git@github.com:mcostalba/Stockfish.git
    pip install python-chess

You can train your own model as:

    python make_data.py -games 1000 | shuf > g1000
    /opt/fastText/fasttext supervised -input g1000 -output model -t 0 -neg 0 -epoch 4

And then run the program as:

    python play_chess.py model.bin
