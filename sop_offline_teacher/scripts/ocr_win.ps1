# Windows 内置 OCR（Windows.Media.Ocr）批量识别截图。
# 必须在 Windows PowerShell 5.1 下运行（pwsh 7 对 WinRT 支持不完整）。
# 用法：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File ocr_win.ps1 -Frames <目录> -Out <输出jsonl>
param(
  [Parameter(Mandatory=$true)][string]$Frames,
  [Parameter(Mandatory=$true)][string]$Out
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime | Out-Null
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile, Windows.Foundation, ContentType=WindowsRuntime]

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
  $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]

function Await($WinRtTask, $ResultType) {
  $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
  $netTask = $asTask.Invoke($null, @($WinRtTask))
  $netTask.Wait(-1) | Out-Null
  $netTask.Result
}

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if (-not $engine) {
  $langs = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages
  Write-Output ("OCR_FAIL 可用语言: " + (($langs | ForEach-Object { $_.LanguageTag }) -join ','))
  exit 2
}
Write-Output ("OCR 语言: " + $engine.RecognizerLanguage.LanguageTag)

$files = Get-ChildItem -Path $Frames -Filter *.png | Sort-Object Name
$sw = [System.IO.StreamWriter]::new($Out, $false, [System.Text.UTF8Encoding]::new($false))
try {
  $i = 0
  foreach ($f in $files) {
    $i++
    try {
      $sf = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($f.FullName)) ([Windows.Storage.StorageFile])
      $stream = Await ($sf.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
      $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
      $bmp = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
      $res = Await ($engine.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
      $lines = @()
      foreach ($ln in $res.Lines) { $lines += $ln.Text }
      $obj = [ordered]@{ file = $f.Name; lines = $lines }
      $sw.WriteLine(($obj | ConvertTo-Json -Compress -Depth 4))
      $stream.Dispose()
    } catch {
      $sw.WriteLine(([ordered]@{ file = $f.Name; error = "$_" } | ConvertTo-Json -Compress))
    }
    if ($i % 20 -eq 0) { Write-Output ("  $i / " + $files.Count) }
  }
} finally {
  $sw.Close()
}
Write-Output ("完成，写入 " + $Out)
