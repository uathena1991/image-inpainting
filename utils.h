typedef struct {
	int x, y;
} Coord;

typedef struct {
	double x, y, z;
} Pixel;

Pixel unfilled_pixel = { 0, 0.9999, 0 };

typedef struct {
	int* x;
	int* y;
	int length;
} Locations;

inline double add(double x, double y);
inline double subtract(double x, double y);
inline double multiply(double x, double y);
inline double divide(double x, double y);
inline double absolute(double x, double y);

inline int equals(double x, double y);
inline int not_equals(double x, double y);
inline int greater_than(double x, double y);
