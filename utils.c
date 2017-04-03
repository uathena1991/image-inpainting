inline double add(double x, double y) {
	return x + y;
}

inline double multiply(double x, double y) {
	return x * y;
}

inline double divide(double x, double y) {
	return x / y;
}

inline double absolute(double x, double y) {
	if (x < 0)
		return multiply(x, -1);
	else
		return x;
}

inline int equals(double x, double y) {
	if (x == y)
		return 1;
	else
		return 0;
}

inline int greater_than(double x, double y) {
	if (x > y)
		return 1;
	else
		return 0;
}
