<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8" />
    <title>Riconoscimento Audio</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        button { font-size: 1.2em; padding: 10px 20px; margin: 10px; cursor: pointer; }
        #output { margin-top: 20px; white-space: pre-wrap; background: #f0f0f0; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Riconoscimento Audio</h1>

    <form method="post">
        <button name="action" value="record">Avvia Registrazione</button>
        <button name="action" value="recognize">Avvia Riconoscimento</button>
    </form>

    <div id="output">
        <?php
        if ($_SERVER["REQUEST_METHOD"] === "POST") {
            $action = $_POST['action'] ?? '';
            if ($action === "record") {
                // Esegui script di registrazione (adatta il path e python se serve)
                $output = shell_exec("python /home/ric/ProgettoIoT/record.py 2>&1");
                echo htmlspecialchars($output);
            }
            elseif ($action === "recognize") {
                echo "Avvio riconoscimento...\n";
                // Esegui script di classificazione (adatta il path)
                $output = shell_exec("python /home/ric/ProgettoIoT/classify.py /home/ric/ProgettoIoT/recording.wav 2>&1");
                echo htmlspecialchars($output);
            }
        }
        ?>
    </div>
</body>
</html>
