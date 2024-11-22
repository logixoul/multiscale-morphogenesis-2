#include "precompiled.h"
#include "colorspaces.h"

vec3 FromHSL(HslF const& hsl)
{
    float v;
    float r, g, b;

	float H = hsl.h, S = hsl.s, L = hsl.l;

    r = L;   // default to gray
    g = L;
    b = L;
    v = (L <= 0.5) ? (L * (1.0 + S)) : (L + S - L * S);
    if (v > 0)
    {
        float m;
        float sv;
        int sextant;
        float fract, vsf, mid1, mid2;

        m = L + L - v;
        sv = (v - m) / v;
        H *= 6.0;
        sextant = (int)H;
        fract = H - sextant;
        vsf = v * sv * fract;
        mid1 = m + vsf;
        mid2 = v - vsf;
        switch (sextant)
        {
            case 0:
                r = v;
                g = mid1;
                b = m;
                break;
            case 1:
                r = mid2;
                g = v;
                b = m;
                break;
            case 2:
                r = m;
                g = v;
                b = mid1;
                break;
            case 3:
                r = m;
                g = mid2;
                b = v;
                break;
            case 4:
                r = mid1;
                g = m;
                b = v;
                break;
            case 5:
                r = v;
                g = m;
                b = mid2;
                break;
        }
    }
    return vec3(r, g, b);
}

HslF::HslF(float h, float s, float l)
{
	this->h = h;
	this->s = s;
	this->l = l;
}

HslF::HslF(ci::vec3 const& rgb)
{
	float r = rgb.x, g = rgb.y, b = rgb.z;
	float max = std::max(r, std::max(g, b)), min = std::min(r, std::min(g, b));
	l = (max + min) / 2;

    if(max == min)
	{
		h = s = 0; // achromatic
	}
	else
	{
		float d = max - min;
		s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
		if(max == r)
			h = (g - b) / d + (g < b ? 6 : 0);
		else if(max == g)
			h = (b - r) / d + 2;
		else if(max == b)
			h = (r - g) / d + 4;
        else
            throw new std::runtime_error("NaN");
		h /= 6;
	}
}
