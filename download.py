# 2) Google Drive 대용량 파일 다운로드용 wget 스크립트
FILE_ID="1a2B3cD4EfGhIjKlMnOpQRsTuVwXyZ0"  # 위에서 복사한 ZIP 파일의 ID
FILENAME="country1.zip"

# 비공식 wget 자동 추출 & 다운로드
wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$(\
    wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
      "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O- \
      | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=${FILE_ID}" \
  -O ${FILENAME} && rm -rf /tmp/cookies.txt

# 3) ZIP 풀기
unzip ${FILENAME}