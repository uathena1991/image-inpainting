#include <math.h>

inline double add(double x, double y) {
	return x + y;
}

inline double subtract(double x, double y) {
	return add(x, -y);
}

inline double multiply(double x, double y) {
	return x * y;
}

inline double divide(double x, double y) {
	return x / y;
}

inline double absolute(double x, double y) {
	if (x < 0)
		return fabs(x);
	else
		return x;
}

inline int equals(double x, double y) {
	return (x == y);
}

inline int not_equals(double x, double y) {
	return ~equals(x, y);
}

inline int greater_than(double x, double y) {
	return (x > y);
}
