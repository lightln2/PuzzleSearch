# Complete Breadth-First Search solver of various puzzles

The solver implements several well known algorithms:
- standard Breadth-First Search
- Disk-Based Frontier Searh
- Two-Bit Breadth-Frst Search
and new algorithms:
- Optimized Frontier Search
- Three-Bit Breadth-First Search
The puzzle can be plugged as an implementation of an interface, and used with all algoritms.

New algorithms incorporate various optimizations:
- data compression
- GPU acceleration
- low-level bit tricks

On my computer, running times are the following:
Fifteen Sliding-Tile Puzzle: 40 hours
Fifteen Pancakes Problem: 62 hours
Four-Peg Towers of Hanoi with 24 disks: 163 hours

Minimum requirements:
- NVidia GPU card
- 8TB free disk space
- 16GB RAM

## License

- **[MIT license](https://lightln2.github.io/PuzzleSearch/license.txt)**
- Copyright 2023 © lightln2
