# PowerShell script for Lab Work H3, all C, S1, O6
# Creates folders and executes all steps for all images

# ---- 0. Create folders for results ----
$folders = @("hist", "hrayleigh", "slowpass", "oll")
foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Name $f
    }
}

# ---- 1. Define all images ----
$images_gray = @{
    "original" = "lena_small.bmp"
    "normal_noise" = "lena_normal2_small.bmp"
    "uniform_noise" = "lena_uniform2_small.bmp"
    "impulse_noise" = "lena_impulse2_small.bmp"
}

$images_color = @{
    "original" = "lenac_small.bmp"
    "normal_noise" = "lenac_normal2_small.bmp"
    "uniform_noise" = "lenac_uniform2_small.bmp"
    "impulse_noise" = "lenac_impulse2_small.bmp"
}

# ---- STEP 1: Histogram calculation ----
Write-Host "`n===== STEP 1: Histogram Calculation =====`n"
foreach ($key in $images_gray.Keys) {
    $input = $images_gray[$key]
    $output = "hist/" + "hist_" + $input + ".png"
    Write-Host "Processing histogram for $input"
    python main.py --histogram -input="$input" -output="$output" -channel=0
}

# ---- STEP 2: Rayleigh Enhancement (H3) ----
Write-Host "`n===== STEP 2: Rayleigh Enhancement =====`n"
foreach ($key in $images_gray.Keys) {
    $input = $images_gray[$key]
    $output = "hrayleigh/" + $input + "_hrayleigh.bmp"
    Write-Host "Applying Rayleigh enhancement to $input"
    python main.py --hrayleigh -input="$input" -output="$output" -alpha=50
}

# ---- STEP 3: Histogram after Rayleigh ----
Write-Host "`n===== STEP 3: Histogram after Rayleigh =====`n"
foreach ($key in $images_gray.Keys) {
    $input = "hrayleigh/" + $images_gray[$key] + "_hrayleigh.bmp"
    $output = "hist/" + "hist_" + $images_gray[$key] + "_hrayleigh.png"
    Write-Host "Processing histogram for $input"
    python main.py --histogram -input="$input" -output="$output" -channel=0
}

# ---- STEP 4: Compute characteristics C1-C6 ----
Write-Host "`n===== STEP 4: Compute Characteristics C1-C6 =====`n"
$characteristics = @("--cmean","--cvariance","--cstdev","--cvarcoi","--casyco","--cflatco","--cvarcoii","--centropy")
foreach ($ch in $characteristics) {
    foreach ($key in $images_gray.Keys) {
        $input_orig = $images_gray[$key]
        $input_rayleigh = "hrayleigh/" + $images_gray[$key] + "_hrayleigh.bmp"

        Write-Host "Running $ch on original $input_orig"
        python main.py $ch -input="$input_orig"

        Write-Host "Running $ch on Rayleigh $input_rayleigh"
        python main.py $ch -input="$input_rayleigh"
    }
}

# ---- STEP 5: Linear filtration S1 (Low-pass) ----
Write-Host "`n===== STEP 5: Low-pass Filtering (S1) =====`n"
foreach ($key in $images_gray.Keys) {
    $input = $images_gray[$key]
    $output = "slowpass/" + $input + "_slowpass.bmp"
    Write-Host "Applying low-pass filter on $input"
    python main.py --slowpass -input="$input" -output="$output"
}

# ---- STEP 6: Non-linear filtration O6 (LL operator) ----
Write-Host "`n===== STEP 6: LL Operator (O6) =====`n"
foreach ($key in $images_gray.Keys) {
    $input = $images_gray[$key]
    $output = "oll/" + $input + "_oll.bmp"
    Write-Host "Applying LL operator on $input"
    python main.py --oll -input="$input" -output="$output" -eps=1e-6
}

# ---- STEP 7: Compute characteristics after S1 and O6 ----
Write-Host "`n===== STEP 7: Characteristics after S1 and O6 =====`n"
foreach ($ch in $characteristics) {
    foreach ($key in $images_gray.Keys) {
        $input_slowpass = "slowpass/" + $images_gray[$key] + "_slowpass.bmp"
        $input_oll = "oll/" + $images_gray[$key] + "_oll.bmp"

        Write-Host "Running $ch on slow-pass $input_slowpass"
        python main.py $ch -input="$input_slowpass"

        Write-Host "Running $ch on LL operator $input_oll"
        python main.py $ch -input="$input_oll"
    }
}

Write-Host "`n===== ALL COMMANDS COMPLETED =====`n"
