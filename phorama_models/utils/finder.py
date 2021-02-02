import os
import mimetypes

class ImageFinder:
    def search(self, directory):
        found = []
        search = [directory]

        while len(search) != 0:
            tmp = [search[0] + '/' + file for file in os.listdir(search[0])]

            for file in tmp:
                mime = mimetypes.guess_type(file)[0]
                
                if mime != None and 'image' in mime:
                    print(file, mime)
                    found.append(file)

                elif not(os.path.isfile(file)):
                    search.append(file)

            del search[0]

        print(len(found))
        return found

        
