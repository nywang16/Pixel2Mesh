construct feed dictionary by pickle load importing the .dat composed by two parts,
pkl[0]: coordinates (name='features')
pkl[1]: support 1
pkl[2]: support 2
pkl[3]: support 3
pkl[4]: pool_idx
pkl[5]: faces
pkl[6]: unknown
pkl[7]: laplacian_normalization

pkl
[0]: coordinates: 
shape: (156, 3), 156 vertices with 3-dim coordinates

[1]: unknown, feed into support1
    [0]:
        [0]: shape (156, 2) int, all composed by [n, n], n from 0 to 155
        [1]: shape (156,), = ones(156, 1) float
        [2]: = [156, 156]
    [1]:
        [0]: shape (1080, 2) int
        [1]: shape (1080,) float
        [2]: = [156, 156]

[2]: unknown, feed into support2
    [0]:
        [0]: shape (618, 2) int, all composed by [n, n], n from 0 to 617
        [1]: shape (156,), = ones(618, 1) float
        [2]: [618, 618]
    [1]: 
        [0]: shape (4314, 2) int
        [1]: shape (4314,) float
        [2]: = [618, 618]

[3]: unknown, feed into support3
    [0]:
        [0]: shape(2466, 2) int, all composed by [n, n], n from 1 to 156
        [1]: shape(2466,) = ones(2466, 1)
        [2]: = [2466, 2466]
    [1]:
        [0]: shape (17250, 2) int
        [1]: shape (17250,) float
        [2]: = [2466, 2466]

[4]: pool_idx
    [0]: shape (462, 2) int, seems like coordinates on 2D
    [1]: shape (1848, 2) int, seems like coordinates on 2D

[5]: faces
    [0]: shape (462, 4) int
    [1]: shape (1848, 4) int
    [2]: shape (7392, 4) int

[6]: unknown
    [0]: shape (156, 3) float. very small number (first block)
    [1]: shape (618, 3) float, very small number (NOT 628) (second block)
    [2]: shape (2466, 4) flaot, very small number (third block)

[7]: laplacian normalization
    [0]: shape (156, 10) int, [-3] and [-4] are always -1 
    [1]: shape (618, 10) int
    [2]: shape (2466, 10) int

