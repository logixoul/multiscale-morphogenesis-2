#include "precompiled.h"
#include "myFFT.h"
#include "gpgpu.h"
#include "stuff.h"

#if 0
static int count;

string compmul = R"END(
		vec2 compmul(vec2 p, vec2 q) {
			return mat2(p,-p.y,p.x) * q;
		}
		)END";

string textureFetchWrap =
"vec4 texelFetchWrap(ivec2 fc) {"
"	if(fc.x >= texSize.x) return vec4(0);"
"	else return texelFetch(tex, fc, 0);"
"}";

// https://rosettacode.org/wiki/Fast_Fourier_transform#C.23

static void fftGpuOneAxis(gl::TextureRef& X, int problemSize, int axis, FFTScalar expSign) {
	static string getWo =
		compmul +
		"vec2 getWo(int indexInSubproblem, vec2 o) {"
		"const float M_PI = 3.14159265359;"
		"float k = float(indexInSubproblem);"
		"float arg = expSign * 2.*M_PI*k / problemSize;"
		"vec2 w = vec2(cos(arg), sin(arg));"
		"return compmul(w, o);"
		"}";

	glBindImageTexture(0, X->getId(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
	shade2(X,
		"ivec2 fragCoord = ivec2(gl_FragCoord.xy);"
		"int indexInSubproblem = fragCoord[axis] % problemSize;"
		"if(indexInSubproblem >= halfProblemSize) {"
		"	discard;"
		"}"
		"ivec2 fetchPosE = fragCoord;"
		"vec2 e = texelFetch(tex, fetchPosE, 0).rg;"
		"ivec2 fetchPosO = fetchPosE;"
		"fetchPosO[axis] += halfProblemSize;"
		"vec2 o = texelFetch(tex, fetchPosO, 0).xy;"
		"vec2 wo = getWo(indexInSubproblem, o);"
		"memoryBarrier();"
		"imageStore(image, fetchPosE, vec4(e + wo, 0, 0));"
		"imageStore(image, fetchPosO, vec4(e - wo, 0, 0));"
		, ShadeOpts().scope("e+wo").enableResult(false).enableWrite(false)
				.uniform("expSign", expSign)
				.uniform("problemSize", problemSize)
				.uniform("halfProblemSize",  problemSize / 2);
				.uniform("axis", axis);
			, getWo
	);
}

void breadthFirst(ivec2 logicalTexSize, gl::TextureRef& in, int axis, FFTScalar expSign) {
	GPU_SCOPE("breadthFirst");
	int N = logicalTexSize[axis];
	int bits = (int)ilog2(N);
	glBindImageTexture(0, in->getId(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
	shade2(in,
		"ivec2 hereCoords = ivec2(gl_FragCoord.xy);"
		"ivec2 thereCoords = hereCoords;"
		"thereCoords[axis] = bitReverse(thereCoords[axis]);"

		"if (thereCoords[axis] <= hereCoords[axis]) return;"
		
		"vec4 fromHere = texelFetch(tex, hereCoords, 0);"
		"vec4 fromThere = texelFetch(tex, thereCoords, 0);"
		"memoryBarrier();"
		"imageStore(image, hereCoords, fromThere);"
		"imageStore(image, thereCoords, fromHere);"
		, ShadeOpts().scope("bitReverse").enableResult(false).enableWrite(false),
		R"END(
		int bitReverse(uint b) {
			b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
			b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
			b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
			b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
			b = ((b >> 16) | (b << 16)) >> (32 - uint(bits)); // todo: rm cast
			return int(b);
		}
		)END"
		, ShadeOpts()
			.uniform("axis", axis)
			.uniform("bits", bits);
	
	);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	for (int problemSize = 2; problemSize <= N; problemSize *= 2) {
		fftGpuOneAxis(in, problemSize, axis, expSign);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}
}

void normalizeResult(gl::TextureRef& in) {
	glBindImageTexture(0, in->getId(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
	shade2(in,
		"ivec2 fragCoord = ivec2(gl_FragCoord.xy);"
		"vec2 loaded = texelFetch(tex, fragCoord, 0).xy;"
		"vec2 toStore = loaded * normMul;"
		"memoryBarrier();"
		"imageStore(image, fragCoord, vec4(toStore, 0, 0));"
		, ShadeOpts().enableResult(false).enableWrite(false)
			.uniform("normMul", 1.0f / sqrt(in->getWidth()*in->getHeight()))
	);
}

gl::TextureRef splitOp(ivec2 resultSize, int N, gl::TextureRef in, gl::TextureRef A, gl::TextureRef B) {
	//cout << "\tin->getSize() = " << in->getSize() << endl;
	//cout << "\tresultSize = " << resultSize << endl;
	setWrap(in, GL_REPEAT);
	gl::TextureRef res = shade2(in, A, B,
		"if(fc.x == int(texSize.x) - 1) {"
		"	ivec2 fcLeftmost = fc;"
		"	fcLeftmost.x = 0;"
		"	vec2 atLeftmost = texelFetch(tex, fcLeftmost, 0).xy;"
		"	_out.x = atLeftmost.x - atLeftmost.y;"
		"	_out.y = 0;"
		"} else {"
		"	int k = fc.x;"
		"	_out.x ="
		"		tF(in, k).x * tF0(A, k).x"
		"		- tF(in, k).y * tF0(A, k).y"
		"		+ tF(in, N / 2 - k).x * tF0(B, k).x"
		"		+ tF(in, N / 2 - k).y * tF0(B, k).y;"
		"	_out.y ="
		"		tF(in, k).y * tF0(A, k).x"
		"		+ tF(in, k).x * tF0(A, k).y"
		"		+ tF(in, N / 2 - k).x * tF0(B, k).y"
		"		- tF(in, N / 2 - k).y * tF0(B, k).x;"
		"}"
		, ShadeOpts()
		.scope("splitOp")
		.dstRectSize(resultSize)  // I do this to have the result tex have these dimensions
		.uniform("N", N)
		,
		"ivec2 fc = ivec2(gl_FragCoord.xy);"
		"vec4 tF(sampler2D s, int x) {"
		"	ivec2 fc2 = fc;"
		"	fc2.x = x;"
		"	vec2 tc_ = (vec2(fc2) + .5) / textureSize(s, 0);"
		"	return textureLod(s, tc_, 0);"
		"}"
		"vec4 tF0(sampler2D s, int x) {"
		"	ivec2 fc2 = fc;"
		"	fc2.x = x;"
		"	fc2.y = 0;"
		"	vec2 tc_ = (vec2(fc2) + .5) / textureSize(s, 0);"
		"	return textureLod(s, tc_, 0);"
		"}"
		"\n#define in tex\n"
		"\n#define A tex2\n"
		"\n#define B tex3\n"
	);

	return res;
}

template<class T>
Array2D<T> garr(vector<T> v) {
	Array2D<T> res(v.size(), 1);
	forxy(res) {
		res(p) = v[p.x];
	}
	return res;
}

void vPushBackV0(gl::TextureRef& v) {
	glBindImageTexture(0, v->getId(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
	shade2(v,
		"ivec2 fc = ivec2(gl_FragCoord.xy);"
		"ivec2 fcLeftmost = fc;"
		"fcLeftmost.x = 0;"
		"vec2 toStore = texelFetch(tex, fcLeftmost, 0).xy;"
		"memoryBarrier();"
		"imageStore(image, fcLeftmost, vec4(toStore, 0, 0));"
		, ShadeOpts()
			.dstPos(ivec2(v->getWidth() - 1, 0))
			.dstRectSize(ivec2(1, v->getHeight()))
			.enableResult(false)
			.enableWrite(false)
	);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void myFFT::fftGpu(gl::TextureRef& in)
{
	//cout << "fft:" << endl;
	beginRTT(nullptr);
	FFTScalar expSign = -1.0;

	//ivec2 outSize = ivec2(in->getWidth() / 2 + 1, in->getHeight());
	
	vector<FFTComplex> Avec, Bvec;
	int N = in->getWidth() * 2;
	getAAndB(N, Avec, Bvec, FFTDir::Forward);
	static auto A = gtex(garr(Avec)); // todo: make a proper cache instead of static
	static auto B = gtex(garr(Bvec));
	auto v = maketex(in->getWidth() + 1, in->getHeight(), GL_RG32F);
	shade2(in,
		"ivec2 fc = ivec2(gl_FragCoord.xy);"
		"fc.x *= 2;"
		"float f1 = texelFetchWrap(fc).x;"
		"float f2 = texelFetchWrap(fc + ivec2(1, 0)).x;"
		"_out.rg = vec2(f1, f2);"
		, ShadeOpts()
			.targetTex(v)
			.dstPos(ivec2())
			.dstRectSize(in->getSize())
			/*.enableResult(false)/*.enableWrite(false)*/
		,
		::textureFetchWrap
	);
	ivec2 inSize = in->getSize();
	in.reset(); TextureCache::clearCaches(); // todo: rm this?
	breadthFirst(inSize, v, 0, expSign);
	vPushBackV0(v);
	v = splitOp(v->getSize(), N, v, A, B);
	breadthFirst(inSize, v, 1, expSign);
	normalizeResult(v);
	in = v;
	endRTT();
}

void myFFT::ifftGpu(gl::TextureRef& in, ivec2 returnSize)
{
	//cout << "ifft:" << endl;
	GPU_SCOPE("ifftGpu");
	beginRTT(nullptr);
	FFTScalar expSign = 1.0;
	
	ivec2 outSize = ivec2((in->getWidth() - 1) * 2, in->getHeight());
	
	breadthFirst(in->getSize(), in, 1, expSign);

	int N = outSize.x;
	vector<FFTComplex> Avec, Bvec;
	getAAndB(N, Avec, Bvec, FFTDir::Backward);
	static auto A = gtex(garr(Avec)); // todo: make a proper cache instead of static
	static auto B = gtex(garr(Bvec));
	in = splitOp(in->getSize() - ivec2(1, 0), N, in, A, B); 
	breadthFirst(in->getSize(), in, 0, expSign);
	in = shade2(in,
		"ivec2 fc = ivec2(gl_FragCoord.xy);"
		"ivec2 fcFetch = fc;"
		"fcFetch.x /= 2;"
		"vec2 f = texelFetch(tex, fcFetch, 0).xy;"
		"if(fc.x % 2 == 0)"
		"	_out.r = f.x;"
		"else"
		"	_out.r = f.y;"
		"_out.r *= normMul;"
		, ShadeOpts()
			.dstRectSize(returnSize) // I do this to have the result tex have these dimensions
			.ifmt(GL_R16F)
			.uniform("normMul", 1.0f / sqrt(in->getWidth()*in->getHeight()));
	);
	
	endRTT();
}
#endif