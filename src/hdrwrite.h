#ifndef _H_RGBE
#define _H_RGBE
/* THIS CODE CARRIES NO GUARANTEE OF USABILITY OR FITNESS FOR ANY PURPOSE.
 * WHILE THE AUTHORS HAVE TRIED TO ENSURE THE PROGRAM WORKS CORRECTLY,
 * IT IS STRICTLY USE AT YOUR OWN RISK.  */

/* utility for reading and writing Ward's rgbe image format.
   See rgbe.txt file for more details.
*/

#include "util.h"

Array2D<vec3> loadRgbeFile(string const& filename);
void writeRgbeFile(string const& filename, ivec2 const& size, float* data);

#endif /* _H_RGBE */


