param (
    [Parameter(Mandatory=$false)]
    [string]$Port = "8000"
)

$ports = $Port -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$allAvailable = $true

foreach ($p in $ports) {
    $portNum = [int]$p
    $listener = $null
    try {
        $ip = [System.Net.IPAddress]::Any
        $listener = New-Object System.Net.Sockets.TcpListener($ip, $portNum)
        $listener.Start()
        $listener.Stop()
        Write-Output "[PORT_OK] Host port $portNum is available."
    } catch {
        Write-Output "[PORT_IN_USE] Host port $portNum is currently in use."
        $allAvailable = $false
    } finally {
        if ($listener -ne $null) {
            try { $listener.Stop() } catch {}
        }
    }
}

if ($allAvailable) {
    exit 0
} else {
    exit 1
}
