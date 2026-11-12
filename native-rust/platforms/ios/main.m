#import <Foundation/Foundation.h>
#import <unistd.h>

extern int voice_changer_ios_main(void);

int main(int argc, char *argv[]) {
    @autoreleasepool {
        // Models can be copied into Documents using Finder file sharing.
        NSURL *documents = [[[NSFileManager defaultManager]
            URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
        chdir(documents.fileSystemRepresentation);
        return voice_changer_ios_main();
    }
}
