#!/bin/sh
# Regenerate the TLS fixtures used by tests/test_http_proxy.py.
#
# The stdlib cannot create certificates, so these files are committed.
# They are test-only: the CA key is thrown away, and nothing else trusts
# this CA. Run from any directory with OpenSSL 1.1.1 or later:
#
#   sh tests/fixtures/tls/generate.sh
#
# Output, valid for 100 years:
#   ca.pem           self-signed test CA
#   localhost.pem    leaf for 127.0.0.1 and localhost, signed by ca.pem
#   wrong-host.pem   leaf for wrong-host.invalid only, signed by ca.pem
#   server.key       private key shared by both leaves
set -eu

cd "$(dirname "$0")"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

days=36500

openssl ecparam -name prime256v1 -genkey -noout -out "$work/ca.key"
openssl req -x509 -new -key "$work/ca.key" -sha256 -days "$days" \
  -subj "/CN=agent-strace test CA" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign" \
  -addext "subjectKeyIdentifier=hash" \
  -out ca.pem

openssl ecparam -name prime256v1 -genkey -noout -out "$work/server.key"
openssl pkcs8 -topk8 -nocrypt -in "$work/server.key" -out server.key

leaf() {
  name=$1
  cn=$2
  san=$3
  cat > "$work/$name.ext" <<EXT
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=serverAuth
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid
subjectAltName=$san
EXT
  openssl req -new -key server.key -subj "/CN=$cn" -out "$work/$name.csr"
  openssl x509 -req -in "$work/$name.csr" -CA ca.pem -CAkey "$work/ca.key" \
    -CAcreateserial -CAserial "$work/ca.srl" -sha256 -days "$days" \
    -extfile "$work/$name.ext" -out "$name.pem"
}

leaf localhost localhost "IP:127.0.0.1,DNS:localhost"
leaf wrong-host wrong-host.invalid "DNS:wrong-host.invalid"
