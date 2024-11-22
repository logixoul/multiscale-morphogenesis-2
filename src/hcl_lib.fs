//imagemagick
// note: returned rgb value is gamma corrected (rather than linear).
vec3 HCL2RGB(vec3 hcl)
{
  float hue = hcl[0], chroma = hcl[1], luma = hcl[2];
  float b,c,g,h,m,r,x,z;

  /*
    Convert HCL to RGB colorspace.
  */
  h=6.0*hue;
  c=chroma;
  x=c*(1.0-abs(mod(h,2.0)-1.0));
  r=0.0;
  g=0.0;
  b=0.0;
  if ((0.0 <= h) && (h < 1.0))
    {
      r=c;
      g=x;
    }
  else
    if ((1.0 <= h) && (h < 2.0))
      {
        r=x;
        g=c;
      }
    else
      if ((2.0 <= h) && (h < 3.0))
        {
          g=c;
          b=x;
        }
      else
        if ((3.0 <= h) && (h < 4.0))
          {
            g=x;
            b=c;
          }
        else
          if ((4.0 <= h) && (h < 5.0))
            {
              r=x;
              b=c;
            }
          else
            if ((5.0 <= h) && (h < 6.0))
              {
                r=c;
                b=x;
              }
  m=luma-(0.298839*r+0.586811*g+0.114350*b);
  /*
    Choose saturation strategy to clip it into the RGB cube; hue and luma are
    preserved and chroma may be changed.
  */
  z=1.0;
  if (m < 0.0)
    {
      z=luma/(luma-m);
      m=0.0;
    }
  else
    if (m+c > 1.0)
      {
        z=(1.0-luma)/(m+c-luma);
        m=1.0-z*c;
      }
		return vec3(z*r+m, z*g+m, z*b+m);
}

// note: expects gamma corrected RGB (rather than linear)
vec3 RGB2HCL(vec3 rgb)
{
  float b,c,g,h,maxval,r;
  float red=rgb.r;
  float green=rgb.g;
  float blue=rgb.b;

  /*
    Convert RGB to HCL colorspace.
  */
  r=rgb.r;
  g=rgb.g;
  b=rgb.b;
  maxval=max(r,max(g,b));
  c=maxval-min(r,min(g,b));
  h=0.0;
  if (c == 0)
    h=0.0;
  else
    if (red == maxval)
      h=mod(6.0+(g-b)/c,6.0);
    else
      if (green == maxval)
        h=((b-r)/c)+2.0;
      else
        if (blue == maxval)
          h=((r-g)/c)+4.0;
  float hue=(h/6.0);
  float chroma=c;
  float luma=0.298839*r+0.586811*g+0.114350*b;
  return vec3(hue, chroma, luma);
}
