// SPDX-License-Identifier: MPL-2.0

Array44i a = Array44i::Random();
cout << "Here is the array a:" << endl << a << endl;
cout << "Here is a.leftCols<2>():" << endl;
cout << a.leftCols<2>() << endl;
a.leftCols<2>().setZero();
cout << "Now the array a is:" << endl << a << endl;
