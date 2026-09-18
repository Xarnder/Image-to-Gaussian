# Junctions comfy_nodes\i2g_export into ComfyUI\custom_nodes so ComfyUI loads the v2 helper nodes.
# Removing the junction (Remove-Item on the link) uninstalls it without touching the source.
. (Join-Path $PSScriptRoot "v2_common.ps1")

$source = Join-Path $RepoRoot "comfy_nodes\i2g_export"
$link = Join-Path $ComfyCustomNodes "i2g_export"
if (Test-Path $link) {
    $item = Get-Item $link
    if ($item.LinkType -eq "Junction" -and $item.Target -contains $source) {
        Write-Log "Junction already in place: $link -> $source"
        exit 0
    }
    throw "$link exists and is not our junction; refusing to overwrite"
}
New-Item -ItemType Junction -Path $link -Target $source | Out-Null
Write-Log "Linked $link -> $source (restart ComfyUI to load it)"
