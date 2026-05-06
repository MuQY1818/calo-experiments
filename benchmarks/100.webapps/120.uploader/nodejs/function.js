const fs = require('fs'),
      path = require('path'),
      request = require('request'),
      storage = require('./storage');

let storage_handler = new storage.storage();

function streamToPromise(stream) {
  return new Promise(function(resolve, reject) {
    stream.on("close", () =>  {
      resolve();
    });
    stream.on("error", reject);
  })
}

exports.handler = async function(event) {
  let bucket = event.bucket.bucket
  let input_prefix = event.bucket.input
  let output_prefix = event.bucket.output
  let object_key = event.object.key
  let url = event.object.url
  let source = object_key || url
  let upload_key = path.basename(source)
  let download_path = path.join(
    '/tmp',
    `${Date.now()}-${Math.random().toString(16).slice(2)}-${upload_key}`
  )

  let promise;
  if (object_key && input_prefix) {
    promise = storage_handler.download(bucket, path.join(input_prefix, object_key), download_path);
  } else {
    var file = fs.createWriteStream(download_path);
    request(url).pipe(file);
    promise = streamToPromise(file);
  }
  var keyName;
  let upload = promise.then(
    async () => {
      [keyName, promise] = storage_handler.upload(bucket, path.join(output_prefix, upload_key), download_path);
      await promise;
      fs.unlinkSync(download_path);
    }
  );
  await upload;
  return {bucket: bucket, url: object_key ? `storage://${bucket}/${path.join(input_prefix, object_key)}` : url, key: keyName}
};
