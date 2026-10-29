; Inno Setup 6 스크립트. build_windows.ps1이 /DAppVersion, /DSourceDir를 넘긴다.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "build\Acappella"
#endif

[Setup]
AppId={{6B1E2B0C-7A1E-4F52-9E1B-3C5A7E11A001}
AppName=아카펠라 채보
AppVersion={#AppVersion}
AppPublisher=동아리 아카펠라 채보
DefaultDirName={localappdata}\Programs\Acappella
DefaultGroupName=아카펠라 채보
PrivilegesRequired=lowest
OutputBaseFilename=AcappellaSetup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
WizardStyle=modern

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "바로가기:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\아카펠라 채보"; Filename: "{app}\Acappella.bat"; WorkingDir: "{app}"
Name: "{autodesktop}\아카펠라 채보"; Filename: "{app}\Acappella.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Acappella.bat"; Description: "지금 실행"; Flags: postinstall nowait skipifsilent shellexec

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python"
