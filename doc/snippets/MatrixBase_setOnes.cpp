// SPDX-License-Identifier: MPL-2.0

Matrix4i m = Matrix4i::Random();
m.row(1).setOnes();
cout << m << endl;
