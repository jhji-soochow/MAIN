# ptflops versions log

## v 0.5.2
- Fix handling of intermediate dimensions in the Linear layer hook.

## v 0.5
- Add per sequential number of parameters estimation.
- Fix sample doesn't work without GPU.
- Clarified output in sample.

## v 0.4
- Allocate temporal blobs on the same device as model's parameters are located.

## v 0.3
- Add 1d operators: batch norm, poolings, convolution.
- Add ability to output extended report to any output stream.

## v 0.2
- Add new operations: Conv3d, BatchNorm3d, MaxPool3d, AvgPool3d, ConvTranspose2d.
- Add some results on widespread models to the README.
- Minor bugfixes.

## v 0.1
- Initial release with basic functionality
