using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.CsProj;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

var benchmark = new SimdBenchmark();
benchmark.Setup();

// var gray = benchmark.SseGray2();
// Image.LoadPixelData<Argb32>(gray, 512, 512)
//     .Save("LennaGray.png");

if (args.Length == 0)
    args = ["--filter", "*"];

BenchmarkSwitcher
    .FromAssembly(typeof(Program).Assembly)
    .Run(args,
        ManualConfig.Create(DefaultConfig.Instance)
            .AddJob(Job.MediumRun
                .WithToolchain(CsProjCoreToolchain.NetCoreApp90))
            .StopOnFirstError());

public class SimdBenchmark
{
    private byte[] pixels;

    [GlobalSetup]
    public void Setup()
    {
        var image = Image.Load<Argb32>("Lenna.png");
        pixels = new byte[image.Width * image.Height * (image.PixelType.BitsPerPixel / 8)];
        image.CopyPixelDataTo(pixels);
    }

    [Benchmark(Baseline = true)]
    public byte[] LinearGray()
    {
        var dest = new byte[pixels.Length];
        for (var i = 0; i < pixels.Length; i += 4)
        {
            var gray = (byte)(pixels[i + 1] * 0.2126 + pixels[i + 2] * 0.7152 + pixels[i + 3] * 0.0722);
            dest[i] = pixels[i];
            dest[i + 1] = gray;
            dest[i + 2] = gray;
            dest[i + 3] = gray;
        }

        return dest;
    }

    [Benchmark]
    public byte[] ParallelGray()
    {
        var dest = new byte[pixels.Length];
        Parallel.For(0, pixels.Length / 4, i =>
        {
            var gray = (byte)(pixels[i * 4 + 1] * 0.2126 + pixels[i * 4 + 2] * 0.7152 + pixels[i * 4 + 3] * 0.0722);
            dest[i * 4] = pixels[i * 4];
            dest[i * 4 + 1] = gray;
            dest[i * 4 + 2] = gray;
            dest[i * 4 + 3] = gray;
        });

        return dest;
    }

    [Benchmark]
    public byte[] SseGray1()
    {
        if (!Sse41.IsSupported)
            throw new Exception();

        var k = Vector128.Create(1.0f, 0.2126f, 0.7152f, 0.0722f);
        var dest = new byte[pixels.Length];

        for (var i = 0; i < pixels.Length; i += 4)
        {
            var v = Vector128.Create((float)pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]);
            var gray = Sse41.DotProduct(v, k, 0b11100001).GetElement(0);

            dest[i] = pixels[i];
            dest[i + 1] = (byte)gray;
            dest[i + 2] = (byte)gray;
            dest[i + 3] = (byte)gray;
        }

        return dest;
    }

    [Benchmark]
    public byte[] SseGray2()
    {
        if (!Sse42.IsSupported)
            throw new Exception();

        var rk = Vector128.Create((short)(0.2126 * short.MaxValue));
        var gk = Vector128.Create((short)(0.7152 * short.MaxValue));
        var bk = Vector128.Create((short)(0.0722 * short.MaxValue));

        var argbMask = Vector128.Create(new byte[] { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 });
        var arMask = Vector128.Create(new byte[] { 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15 });
        var resultMask = Vector128.Create(new byte[] { 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 });
        var r1Mask = Vector128.Create(new byte[] { 0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7 });
        var r2Mask = Vector128.Create(new byte[] { 8, 9, 9, 9, 10, 11, 11, 11, 12, 13, 13, 13, 14, 15, 15, 15 });
        var dest = new byte[pixels.Length];

        var i = 0;
        for (; i < pixels.Length / 32; i += 32)
        {
            // ARBG -> AAAARRRRGGGGBBBB
            var v1 = Vector128.Shuffle(Vector128.Create(pixels, i), argbMask);
            var v2 = Vector128.Shuffle(Vector128.Create(pixels, i + 16), argbMask);

            var l1 = Sse2.UnpackLow(v1, Vector128<byte>.Zero).AsInt16();
            var h1 = Sse2.UnpackHigh(v1, Vector128<byte>.Zero).AsInt16();
            var l2 = Sse2.UnpackLow(v2, Vector128<byte>.Zero).AsInt16();
            var h2 = Sse2.UnpackHigh(v2, Vector128<byte>.Zero).AsInt16();

            // AAAARRRR
            var t1 = Vector128.Shuffle(Sse2.PackUnsignedSaturate(l1, l2), arMask);
            // GGGGBBBB
            var t2 = Vector128.Shuffle(Sse2.PackUnsignedSaturate(h1, h2), arMask);

            var a = Sse2.UnpackLow(t1, Vector128<byte>.Zero).AsInt16();
            var r = Ssse3.MultiplyHighRoundScale(Sse2.UnpackHigh(t1, Vector128<byte>.Zero).AsInt16(), rk);
            var g = Ssse3.MultiplyHighRoundScale(Sse2.UnpackLow(t2, Vector128<byte>.Zero).AsInt16(), gk);
            var b = Ssse3.MultiplyHighRoundScale(Sse2.UnpackHigh(t2, Vector128<byte>.Zero).AsInt16(), bk);
            var gray = Sse2.AddSaturate(r, Sse2.AddSaturate(g, b));

            var result = Vector128.Shuffle(Sse2.PackUnsignedSaturate(a, gray), resultMask);
            var r1 = Vector128.Shuffle(result, r1Mask);
            var r2 = Vector128.Shuffle(result, r2Mask);

            r1.CopyTo(dest, i);
            r2.CopyTo(dest, i + 16);
        }

        for (; i < pixels.Length; i += 4)
        {
            var gray = (byte)(pixels[i + 1] * 0.2126 + pixels[i + 2] * 0.7152 + pixels[i + 3] * 0.0722);
            dest[i] = pixels[i];
            dest[i + 1] = gray;
            dest[i + 2] = gray;
            dest[i + 3] = gray;
        }

        return dest;
    }
}